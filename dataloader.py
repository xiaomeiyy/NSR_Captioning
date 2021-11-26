from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL
from PIL import Image
import json
import h5py
import os
import numpy as np
import random
from functools import reduce

import torch
import torch.utils.data as data

import multiprocessing
import utils
import pickle


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word
    def get_vocab_new(self):
        return self.ix_to_word_new
    def get_vocab_glove(self):
        return self.vocab_dict_glove
    def get_obj_glove(self):
        return self.obj_glove

    def get_seq_length(self):
        return self.seq_length
    
    def rel_word(self, idx_to_rel):
        ## change vocabulary from word to ix
        word_to_ix = {}
        for ix in self.ix_to_word:
            word_to_ix[self.ix_to_word[ix]] = ix
        
        # get max lenght of relations
        max_len_rel = 0
        for idx in idx_to_rel:
            rel = idx_to_rel[idx].split(' ')
            if max_len_rel < len(rel):
                max_len_rel = len(rel)
            
        # get relation words index
        rel_words = np.zeros([len(idx_to_rel)+1, max_len_rel], dtype='int')
        for idx in idx_to_rel:
            rel = idx_to_rel[idx].split(' ')
            for i in range(len(rel)):
                rel_words[int(idx),i] = word_to_ix[rel[i]]
     
        return rel_words
                

    def __init__(self, opt, transform=None):
        self.opt = opt
        self.batch_size = self.opt.TRAIN.BATCH_SIZE
        self.seq_per_img = opt.seq_per_img
        self.transform = transform
        print('batch_size is {}, beam_size is {}'.format(self.batch_size, opt.beam_size))

        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = pickle.load(open(self.opt.input_json,'rb'))
        self.ix_to_word = self.info['ix_to_word']
        self.ix_to_word_new = self.info['ix_to_word_new']
        self.vocab_dict_glove = self.info['vocab_dic_glove'].astype('float32')
        self.vocab_size = len(self.ix_to_word)

        print('vocab size is ', self.vocab_size)
        
        # load the pickle file which contains object glove vectors
        print('DataLoader objects glove matrix from pkl file: ', opt.Obj_glove_file)
        self.obj_glove = pickle.load(open(self.opt.Obj_glove_file,'rb')).astype('float32')
        
        # load relation words from visual geonome dataset
        rel_info = json.load(open(opt.VG_SGG_dictsFile))
        rel_to_idx = rel_info['predicate_to_idx']
        idx_to_rel = rel_info['idx_to_predicate']
        self.rel_dic = self.rel_word(idx_to_rel)        

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
#         self.input_att_dir = self.opt.input_att_dir
        self.input_rel_dir = self.opt.input_rel_dir
        # self.input_box_dir = self.opt.input_box_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        split_noc = json.load(open(self.opt.split_nocfile))
        self.split_ix = {'train': [], 'val': [], 'test': []}
        if self.opt.overfit_train:
            i = 0
            for ix in range(len(self.info['images'])):
                img = self.info['images'][ix]
                if str(img['id']) in split_noc['test']:
                    self.split_ix['train'].append(ix)
                    self.split_ix['test'].append(ix)
                    i += 1
                if i >= 1000:
                    break
        else:
            for ix in range(len(self.info['images'])):
                img = self.info['images'][ix]
                if self.opt.split_noc:                    
                    if str(img['id']) in split_noc['train']:
                        self.split_ix['train'].append(ix)
                    elif str(img['id']) in split_noc['test']:
                        self.split_ix['test'].append(ix)
                else:
                    if img['split'] == 'train':
                        self.split_ix['train'].append(ix)
                    elif img['split'] == 'test':
                        self.split_ix['test'].append(ix)               
                    else:  # restval
                        self.split_ix['train'].append(ix)
        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        img_batch = []
        fc_batch = []  # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')
        box_logits_batch = []
#         box_features_batch = []
#         att_relindx_batch = []
        att_relscore_batch = []

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, ix, tmp_box_logits, att_relscore,tmp_wrapped = self._prefetch_process[split].get()
#             print(att_boxes.shape)
            # img_batch.append(tmp_img)
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            box_logits_batch.append(tmp_box_logits)
#             box_features_batch.append(tmp_box_features)
#             att_relindx_batch.append(att_relindx)
            att_relscore_batch.append(att_relscore)

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1] = self.get_captions(ix, seq_per_img)
            
            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)
            # import matplotlib.pyplot as plt
            # import matplotlib.image as mping
            # img = mping.imread(os.path.join(self.opt.IMAGE_DIR, self.info['images'][ix]['file_path']))
            # imgplot = plt.imshow(img)
            # plt.show()
            # sents = utils.decode_sequence(self.ix_to_word, label_batch)
            # print(sents)

        # #sort by att_feat length
        fc_batch, att_batch, label_batch, gts, infos, box_logits_batch, att_relscore_batch= \
            zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos, box_logits_batch, att_relscore_batch), key=lambda x: 0, reverse=True))
        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] * seq_per_img for _ in fc_batch]))
        # merge att_feats: get the max length of relations/eges and nodes
        max_att_len = max([_.shape[0] for _ in att_batch])
        
        ## initial node
        data['box_logits'] = np.zeros([len(box_logits_batch) * seq_per_img, max_att_len, box_logits_batch[0].shape[-1]],dtype='float32')
#         data['box_features'] = np.zeros([len(box_features_batch) * seq_per_img, max_box_len, box_features_batch[0].shape[-1]],dtype='float32')
        
        # merge att_feats: initialize data
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],
                                     dtype='float32')
#         data['att_relindx'] = np.zeros([len(att_relindx_batch) * seq_per_img, max_attrel_len, att_relindx_batch[0].shape[1]],dtype='float32')
        data['att_relscore'] = np.zeros([len(att_relscore_batch) * seq_per_img, max_att_len, max_att_len, att_relscore_batch[0].shape[-1]],dtype='float32')
        
        for i in range(len(att_batch)):
            data['box_logits'][i * seq_per_img:(i + 1) * seq_per_img, :box_logits_batch[i].shape[0]] = box_logits_batch[i]
#             data['box_features'][i * seq_per_img:(i + 1) * seq_per_img, :box_features_batch[i].shape[0]] = box_features_batch[i]
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
#             data['att_relindx'][i * seq_per_img:(i + 1) * seq_per_img, :att_relindx_batch[i].shape[0]] = att_relindx_batch[i]+1
            data['att_relscore'][i * seq_per_img:(i + 1) * seq_per_img, :att_relscore_batch[i].shape[0], :att_relscore_batch[i].shape[0]] = att_relscore_batch[i]
            
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1

        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]

        ## bottom up boxes and features by myself testing: image_h, features, image_w, img_id, boxscores, boxes, imgscales
        try:
            up_box_feat_info = pickle.load(open(os.path.join(self.opt.UP_box_dir, str(self.info['images'][ix]['id']) + '.pkl'),'rb'),encoding='latin1')
        except:
            print('----------------------------image id {}---------------------'.format(self.info['images'][ix]['id']))
        up_feats = up_box_feat_info['features'] # n * 2048
        up_boxscores = up_box_feat_info['boxscores'] # n * 1601
        up_boxes = up_box_feat_info['boxes'] # n * 4        
        
        ## load relation info
        num_nodes = up_feats.shape[0]   
        att_relscore = np.load(os.path.join(self.input_rel_dir, str(self.info['images'][ix]['id']) + '.npy'))
        rel_score_new = np.reshape(att_relscore,(num_nodes, num_nodes,-1))

        fc_feats = np.mean(up_feats,axis=0)

        return (fc_feats, up_feats, ix, up_boxscores, rel_score_new)

    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=4,  # 4 is usually enough
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]
