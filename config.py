import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = ''
__C.EMBEDDING_TYPE = ''
__C.CONFIG_NAME = ''
__C.CUDA = True
__C.WORKERS = 6

__C.NET_G = ''
__C.NET_D = ''
__C.STAGE1_G = ''
__C.IMAGE_DIR = ''
__C.VIS_COUNT = 64

__C.Z_DIM = 100
__C.IMSIZE = 64
__C.STAGE = 1


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 100
__C.TRAIN.SNAPSHOT_INTERVAL = 3
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 5
__C.TRAIN.LR_DECAY_EPOCH = 5

__C.TEXT = edict()
__C.TEXT.DIMENSION = 1024

# img caption options
__C.drop_prob_lm=0.5
__C.grad_clip = 0.1

__C.beam_size = 1
__C.seq_per_img = 5
__C.fc_feat_size = 2048
__C.att_feat_size = 2048
__C.att_hid_size = 512
__C.rnn_size = 512
__C.input_encoding_size=512

__C.id = 'cap_model_name'
__C.id_model = 'test_model'
__C.cached_tokens = 'coco-train-idxs'
__C.input_att_dir = '/home/wxm/DATASET/cocoK_fea/cocobu/cocobu_att/'
__C.input_fc_dir = '/home/wxm/DATASET/cocoK_fea/cocobu/cocobu_fc'
__C.input_rel_dir = '/home/wxm/DATASET/cocoK_fea/cocobu/cocobu_rel'
__C.input_json = '/home/wxm/DATASET/cocoK_fea/cocotalk.json'
__C.input_label_h5 = '/home/wxm/DATASET/cocoK_fea/cocotalk_label.h5'
__C.annFile = '/home/wxm/exp_wxm/ImageCaption_P1/tools/coco_caption/annotations/captions_val2014.json'
__C.VG_SGG_dictsFile = '/home/wxm/DATASET/visual_genome/VG-SGG-dicts.json'
__C.UP_box_dir = '/home/wxm/DATASET/MSCOCO2014/sv_wxmboxes/'
__C.Obj_glove_file = '/home/wxm/DATASET/MSCOCO2014/object1600_glove.pkl/'
__C.caption_model = 'deeplstm'
__C.vocab_size = 0
__C.num_layers = 1
__C.language_eval = 1
__C.self_critical_after = 30

__C.optim = 'adam'
__C.optim_alpha = 0.9
__C.optim_beta = 0.999
__C.optim_epsilon = 1e-8
__C.weight_decay = 0
__C.cider_reward_weight = 1
__C.bleu_reward_weight = 0

__C.learning_rate = 5e-4
__C.learning_rate_decay_start =3
__C.scheduled_sampling_start = 5
__C.learning_rate_decay_every = 5
__C.learning_rate_decay_rate = 0.8
__C.scheduled_sampling_increase_every = 70
__C.scheduled_sampling_increase_prob = 0.05
__C.scheduled_sampling_max_prob = 0.25
__C.overfit_train = False

__C.label_smoothing = 0
__C.noamopt = False

__C.seed = 1

__C.split_nocfile = '/home/xiaomeiwang/...'
__C.split_noc = False
__C.top_num = 40

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
