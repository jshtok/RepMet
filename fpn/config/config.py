# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong, Bin Xiao
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()
config.test_idx=0
config.MXNET_VERSION = ''
config.output_path = ''
config.symbol = ''
config.gpus = ''
config.CLASS_AGNOSTIC = True
config.SCALES = [(600, 1000)]  # first is scale (the shorter side); second is max size
config.TEST_SCALES = [(600, 1000)]
# default training
config.default = edict()
config.default.frequent = 20
config.default.kvstore = 'device'

# network related params
config.network = edict()
config.network.pretrained = ''
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.IMAGE_STRIDE = 0
config.network.RPN_FEAT_STRIDE = 16
config.network.RCNN_FEAT_STRIDE = 16
config.network.FIXED_PARAMS = ['gamma', 'beta']
config.network.FIXED_PARAMS_SHARED = ['gamma', 'beta']
config.network.FORCE_RESHAPE_EVERY_BATCH = False
config.network.ANCHOR_SCALES = (8, 16, 32)
config.network.ANCHOR_RATIOS = (0.5, 1, 2)
config.network.NUM_ANCHORS = len(config.network.ANCHOR_SCALES) * len(config.network.ANCHOR_RATIOS)
config.network.ALT_FIXED_PARAMS =[]
# Leonid: additional network params for oneshot, config.network.EMBEDDING_DIM = 0 means disabled

config.network.store_kmeans_reps = False # during training, compute kmeans representatives and store the model
config.network.model_kmeans_fname =''
config.network.REPS_DIM = 0 # raw dimension of representatives
config.network.EMBEDDING_DIM = 0
config.network.REPS_PER_CLASS = 30
config.network.SIGMA = 0.2
config.network.EMBED_LOSS_ENABLED = True
config.network.EMBED_LOSS_MARGIN = 0.5
config.network.SOFTMAX_ENABLED = True
config.network.SOFTMAX_MUL = 15.0
config.network.EMBED_L2_NORM = True
config.network.REP_L2_NORM = True
config.network.SEPARABLE_REPS = False
config.network.BG_REPS = False
config.network.BG_REPS_NUMBER = 20
config.network.VISUALIZE_REPS = False
config.network.SMOOTH_MIN = False
config.network.SMOOTH_CONST = -2
config.network.EMBED_LOSS_NEG = True # subtract min_dist_false in embed_loss
config.network.compute_rep_stats = False
config.network.REPS_CLS_LOSS = False
config.network.SEPARABLE_REPS_INIT = False
config.network.ADDITIONAL_LINEAR_CLS_LOSS = False
config.network.NUM_ROIS_FOR_VAL_TRAIN = 100
config.network.VAL_FILTER_REGRESS = False
config.network.SCORE_HIST_REGRESS = False
config.network.SCORE_HIST_TRUNCATE = 0
config.network.SCORE_HIST_WEIGHT = 0 # weight methods: 0 - none. 1 - by sum of (truncated) hist.
config.network.pretrained_weights_are_priority = False
config.network.base_net_lock = False
config.network.EMBED_LOSS_POS= False # use only the positive part of EMBED_LOSS if true
config.network.DEEP_REPS= False # make representatives
config.network.NO_BG_REPULSION = False # if True, fix the participation of background rois in triplet loss
config.network.EMBED_LOSS_GRAD_SCALE = 1.0
config.network.BBOX_LOSS_GRAD_SCALE = 1.0
config.network.CLS_SCORE_GRAD_SCALE = 1.0
config.network.REPS_CLS_LOSS_GRAD_SCALE = 1.0
config.network.ADDITIONAL_LINEAR_CLS_LOSS_GRAD_SCALE = 1.0

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'PascalVOC'
config.dataset.image_set = '2007_trainval'
config.dataset.test_image_set = '2007_test'
config.dataset.root_path = './data'
config.dataset.dataset_path = './data/VOCdevkit'
#config.dataset.databases_csv=''
config.dataset.NUM_CLASSES = 21
config.dataset.per_category_epoch_max = '0'

# Leonid: additional dataset params
config.dataset.order_classes_incrementally = False
config.dataset.balance_classes = False
config.dataset.max_num_extra_classes = 100
config.dataset.num_ex_per_class = 200
config.dataset.num_ex_between_extras = 1
config.dataset.num_ex_base_limit = 0 # 0 means disabled
config.dataset.cls_filter_files = None
config.dataset.filter_roidb = True
config.dataset.homog_data_fname =''

config.TRAIN = edict()
# Leonid: additional training params
config.TRAIN.UPDATE_REPS_VIA_CLUSTERING = False
config.TRAIN.UPDATE_REPS_STOP_EPOCH = 1000
config.TRAIN.UPDATE_REPS_START_EPOCH = 0

config.TRAIN.MAX_CLUSTERING_EPOCH = 100
config.TRAIN.NUMEX_FOR_CLUSTERING = 100
config.TRAIN.REPS_LR_MULT = 1
config.TRAIN.REPS_WD_MULT = 0
config.TRAIN.STORE_DEBUG_IMAGES_EPOCH = 10000

config.TRAIN.lr = 0
config.TRAIN.lr_step = ''
config.TRAIN.lr_factor = 0.1
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = ''
config.TRAIN.RM_LAST = 0 #remove last class from foreground (don`t care objects)
config.TRAIN.ALTERNATE = edict()
config.TRAIN.ALTERNATE.RPN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.RCNN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.rpn1_lr = 0
config.TRAIN.ALTERNATE.rpn1_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn1_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn1_lr = 0
config.TRAIN.ALTERNATE.rfcn1_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn1_epoch = 0      # recommend 8
config.TRAIN.ALTERNATE.rpn2_lr = 0
config.TRAIN.ALTERNATE.rpn2_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn2_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn2_lr = 0
config.TRAIN.ALTERNATE.rfcn2_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn2_epoch = 0      # recommend 8
# optional
config.TRAIN.ALTERNATE.rpn3_lr = 0
config.TRAIN.ALTERNATE.rpn3_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn3_epoch = 0       # recommend 3

# whether resume training
config.TRAIN.RESUME = False
# whether load externally trained embedding layers
config.TRAIN.LOAD_EMBEDDING = False
config.TRAIN.EMBEDDING_FNAME = ''
# whether flip image
config.TRAIN.FLIP = True
# whether shuffle image
config.TRAIN.SHUFFLE = True
# whether use OHEM
config.TRAIN.ENABLE_OHEM = False
# size of images for each device, 2 for rcnn, 1 for rpn and e2e
config.TRAIN.BATCH_IMAGES = 2
# e2e changes behavior of anchor loader and metric
config.TRAIN.END2END = False
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = 128
config.TRAIN.BATCH_ROIS_OHEM = 128
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# used for end2end training
# RPN proposal
config.TRAIN.CXX_PROPOSAL = True
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
config.TRAIN.RPN_POST_NMS_TOP_N = 2000
config.TRAIN.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE
# approximate bounding box regression
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)

config.TEST = edict()

# R-CNN testing
# use rpn to generate proposal
config.TEST.HAS_RPN = False
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.CXX_PROPOSAL = True
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = config.network.RPN_FEAT_STRIDE

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.max_per_image = 300

# Test Model Epoch
config.TEST.test_epoch = 0

config.TEST.USE_SOFTNMS = False


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.SafeLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")
