import numpy as np
import json
import os

import torch
from torch.autograd import Variable

import utils


def language_eval(preds, cache_name, split, annFile):
    import sys
    sys.path.append("/apdcephfs/private_forestlma/xiaomeiwang/exp_wxm/tools/coco_caption")
    # annFile = '/home/wxm/exp_wxm/ImageCaption_P1/tools/coco_caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', cache_name + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def eval_split(model, loader, opt):
    
    split = 'test'
    # change easydict to dict and deleted keywords of opt that is not used for captioning
    opt = dict(opt)
    cache_name = opt['DATASET_NAME'] + '_' + opt['id_model']
    del_GAN_str = []
    for key_word in opt:
        if key_word.isupper():
            del_GAN_str.append(key_word)
    for w in del_GAN_str:
        del opt[w]
    model.eval()
    loader.reset_iterator(split)
    
    predictions = []
    n = 0
    num_images = len(loader.split_ix[split])
    while True:
        data = loader.get_batch(split)
        batch_size = loader.batch_size
        n = n + loader.batch_size

        tmp = [data['fc_feats'][np.arange(batch_size) * opt['seq_per_img']],
               data['att_feats'][np.arange(batch_size) * opt['seq_per_img']],
               data['att_masks'][np.arange(batch_size) * opt['seq_per_img']]
               if data['att_masks'] is not None else None,
               data['box_logits'][np.arange(batch_size) * opt['seq_per_img']],
#                data['box_features'][np.arange(batch_size) * opt['seq_per_img']],
#                data['att_relindx'][np.arange(batch_size) * opt['seq_per_img']],
               data['att_relscore'][np.arange(batch_size) * opt['seq_per_img']]]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks,box_logits, att_relscore = tmp
       
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, box_logits,att_relscore, att_masks, opt, mode='sample')[0].data
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            # if eval_kwargs.get('dump_path', 0) == 1:
            #     entry['file_name'] = data['infos'][k]['file_path']

            # if eval_kwargs.get('dump_images', 0) == 1:
            #     # dump the raw image to vis/ folder
            #     cmd = 'cp "' + os.path.join(eval_kwargs['image_root'],
            #                                 data['infos'][k]['file_path']) + '" vis/imgs/img' + str(
            #         len(predictions)) + '.jpg'  # bit gross
            #     print(cmd)
            #     os.system(cmd)
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        if num_images >= 0 and n >= num_images:
            break
        if data['bounds']['wrapped']:
            break

    annFile = opt['annFile']
    
    lang_stats = language_eval(predictions, cache_name, split, annFile)

    model.train()

    return predictions, lang_stats

