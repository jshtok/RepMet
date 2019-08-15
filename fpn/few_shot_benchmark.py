#------------------------------------------------------------------
# RepMet few-shot detection engine
# Copyright (c) 2019 IBM Corp.
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Joseph Shtok, josephs@il.ibm.com, Leonid Karlinsky, leonidka@il.ibm.com, IBM Research AI, Haifa, Israel
#------------------------------------------------------------------

import _init_paths
import argparse
import sys
import os
sys.path.append("/dccstor/jsdata1/dev/RepMet/lib")
import cv2
import mxnet as mx
import numpy as np
from utils.image import resize, transform
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from config.config import config, update_config
from core.tester import im_detect,im_detect_feats, im_detect_feats_stats, Predictor
from nms.nms import gpu_nms_wrapper
from symbols import *
from data_hub import load_a_model, names_list_to_indices,get_train_objects_fname
import cPickle
from utils.miscellaneous import print_img,assert_folder,configure_logging,flatten, PerfStats,BWlists,compute_det_types
import copy
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
import time
random.seed(9001)
from random import random as rand

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
data_names = ['data', 'im_info']
root = '/dccstor/jsdata1/dev/RepMet_project/RepMet_CVPR19'

# ------------------------------------------------------------------------------------------------------


def bb_overlap(bb_array, bb_GT):
    # bb_GT: [Ninstances,4] matrix
    # bb_array: [Nrois, 4]
    # for every row
    if bb_GT.ndim == 1:
        bb_GT = np.expand_dims(bb_GT,0)
    overlaps = np.zeros((bb_array.shape[0], bb_GT.shape[0]), np.float32)  # [Nrois, Ninstances]
    for i,GT in enumerate(bb_GT):  # go over rows
        # intersection
        ixmin = np.maximum(bb_array[:, 0], GT[0])  # [Nrois, 1]
        iymin = np.maximum(bb_array[:, 1], GT[1])
        ixmax = np.minimum(bb_array[:, 2], GT[2])
        iymax = np.minimum(bb_array[:, 3], GT[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih # [Nrois, 1]

        # union
        uni = ((GT[2] - GT[0] + 1.) * (GT[3] - GT[1] + 1.) +
               (bb_array[:, 2] - bb_array[:, 0] + 1.) *
               (bb_array[:, 3] - bb_array[:, 1] + 1.) - inters)

        overlaps[:, i] = inters / uni

    return overlaps

def prep_data_single(im_path):
    data = []
    if type(im_path) == str:
        assert os.path.exists(im_path), ('%s does not exist'.format(im_path))
        im = cv2.imread(im_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        im = im_path
    target_size = config.SCALES[0][0]
    max_size = config.SCALES[0][1]
    im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
    im_tensor = transform(im, config.network.PIXEL_MEANS)
    im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
    data.append({'data': im_tensor, 'im_info': im_info})
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                 provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                 provide_label=[None])
    return data, data_batch, im_scale, im

def get_ovp_rois(roi_targ, rois_embed, overlap_thresh, do_nms=False):
    bb_indices = []
    cat_ids = []
    for GT_idx, gtbb in enumerate(roi_targ):
        overlaps = bb_overlap(rois_embed[:,1:],gtbb[0:-1])
        inst_indices = np.where(overlaps>=overlap_thresh)[0]
        if len(inst_indices)==0:
            print('Box #{0} - no overlapping detection boxes found'.format(GT_idx))
        else:
            a = zip(*sorted(zip(overlaps[inst_indices], inst_indices)))
            inst_indices = np.asarray(a[1])
            if do_nms:
                scores_FG = overlaps[inst_indices]
                boxes_FG = rois_embed[inst_indices,1:]
                cls_dets = np.hstack((boxes_FG, scores_FG))
                keep = nms_ovp_rois(cls_dets)
                inst_indices = inst_indices[keep]
            inst_indices = inst_indices.tolist()
            #print('Box #{0} - max. overplap:{1:.4f}'.format(GT_idx,np.max(overlaps[inst_indices])))
            bb_indices.extend(inst_indices)
            cat_ids.extend([gtbb[-1]]*len(inst_indices))
    return bb_indices, cat_ids

def extract_embeds(psp_embed, bb_indices):
    # psp_embed.shape = [#rois, embed_dim]
    # bb_indices - indices of the rois produced by the model that overlap well with GT bboxes
    # custom_embed = 7x7 array of 2d arrays of dimensions [EMBED_DIM, #selected_rois], gathered from image i.

    psp_embed_chunk = np.squeeze(psp_embed)  # rois, embed_dim]
    psp_embed_chunk = psp_embed_chunk[bb_indices, :]  # [rois, embed_dim]
    custom_embed = np.transpose(psp_embed_chunk, (1, 0))
    return custom_embed

def compute_tot_embeds(cat_ids, n_cats, custom_embed, use_kmeans=True):

    from sklearn.cluster import KMeans

    n_clusters = config.network.REPS_PER_CLASS
    tot_embeds = [None] * (n_cats)
    for iCls in range(n_cats):  # range(max_cls):
        if iCls in cat_ids:
            samples = np.stack([custom_embed[:, i] for i, c in enumerate(cat_ids) if c == iCls], axis=1)
            if (samples.size > 0) and (samples.shape[1] >= n_clusters):
                if use_kmeans:
                    kmeans = KMeans(n_clusters).fit(samples.T)
                    clst = kmeans.cluster_centers_.T
                else:
                    clst = samples[:,0:n_clusters]
            else:
                clst = np.concatenate([samples,1000 * np.ones((config.network.EMBEDDING_DIM, n_clusters-samples.shape[1]))],axis=1)
        else:
            clst = 1000 * np.ones((config.network.EMBEDDING_DIM, n_clusters))
        tot_embeds[iCls] = clst
    return tot_embeds

def add_reps_to_model(arg_params, tot_embeds, Nreplaced=0, from_start=False):
    # tot_embeds[i] is a cat_embeds for a category i. cat_embeds = [embed_dim, #REPS_PER_CAT]
    # by default, new representatives are added with new class indices at the end of the  class list.
    # if from_start, then new representatives replace old classes from the beginning of the class list.

    new_reps = []
    reps = arg_params['fc_representatives_weight']  # shape=(cfg.network.EMBEDDING_DIM*cfg.network.REPS_PER_CLASS*(num_classes-1))
    Nclasses = reps.shape[0] / (config.network.EMBEDDING_DIM * config.network.REPS_PER_CLASS) + 1
    reps = mx.nd.reshape(reps, shape=(config.network.EMBEDDING_DIM, config.network.REPS_PER_CLASS, Nclasses-1))

    for icat, cat_embeds in enumerate(tot_embeds):
        new_reps = np.expand_dims(cat_embeds, axis=2)
        if Nreplaced>0:
            reps = reps[:,:,:-Nreplaced]
        if from_start:
            new_reps = np.expand_dims(cat_embeds, axis=2)
            reps[:, :, icat] = mx.nd.array(new_reps)
        else:
            reps = mx.nd.concat(reps, mx.nd.array(new_reps), dim=2)
            Nclasses+=1
    reps = mx.nd.reshape(reps, (config.network.EMBEDDING_DIM * config.network.REPS_PER_CLASS * (Nclasses-1), 1))
    arg_params['fc_representatives_weight'] = reps
    return arg_params, new_reps, Nclasses

def random_transform(image, boxes, image_data_generator,  seed=None,cats_img=None):
    if seed is None:
        seed = np.random.randint(10000)

    # leonid: added this to support any augmentation including those changing image size
    orig_image_shape=image.shape

    image = image_data_generator.random_transform(image, seed=seed)

    # set fill mode so that masks are not enlarged
    fill_mode = image_data_generator.fill_mode
    image_data_generator.fill_mode = 'constant'

    invalid_indices=[]
    for index in range(boxes.shape[0]):
        # generate box mask and randomly transform it
        mask = np.zeros(shape=orig_image_shape, dtype=np.uint8)
        b = boxes[index, :4].astype(int)
        cv2.rectangle(mask, (b[0], b[1]), (b[2], b[3]), (255,) * image.shape[-1], -1)
        mask = image_data_generator.random_transform(mask, seed=seed)[..., 0]
        mask = mask.copy()  # to force contiguous arrays

        # find bounding box again in augmented image
        [i, j] = np.where(mask == 255)
        if len(i)>0:
            boxes[index, 0] = float(min(j))
            boxes[index, 1] = float(min(i))
            boxes[index, 2] = float(max(j))
            boxes[index, 3] = float(max(i))
        else:
            invalid_indices.append(index)

    # delete boxes that went out of the image
    boxes = np.delete(boxes, invalid_indices, 0)
    if cats_img is not None:
        cats_img = np.delete(cats_img,invalid_indices,0)
    # restore fill_mode
    image_data_generator.fill_mode = fill_mode
    if cats_img is not None:
        return image, boxes, cats_img
    else:
        return image, boxes

def show_boxes_loc(im, img_train_bbs, class_inds, scale = 1.0, save_file_path='temp.png'):
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for bb in img_train_bbs:
        bbox = bb[:4] * scale
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        # print scores: ------------------------
        # if cls_dets.shape[1] == 5:
        #     score = det[-1]
        #     plt.gca().text(bbox[0], bbox[1],
        #                    '{:s} {:.3f}'.format(cls_name, score),
        #                    bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    fig.savefig(save_file_path)

def train_net(sym_instance, ctx, roidb, arg_params, aux_params, begin_epoch=0, end_epoch=10, lr=0.001, lr_step='4,6,20'):
    # mx.random.seed(3)
    # np.random.seed(3)

    # load symbol
    sym = sym_instance.get_symbol(config, is_train=True)

    feat_pyramid_level = np.log2(config.network.RPN_FEAT_STRIDE).astype(int)
    feat_sym = [sym.get_internals()['rpn_cls_score_p' + str(x) + '_output'] for x in feat_pyramid_level]

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = min(config.TRAIN.BATCH_IMAGES * batch_size,len(roidb))

    # load training data
    from core.loader import PyramidAnchorIterator
    train_data = PyramidAnchorIterator(feat_sym, roidb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE,
                                       ctx=ctx, feat_strides=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
                                       anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
                                       allowed_border=np.inf)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))
    # print 'providing maximum shape', max_data_shape, max_label_shape

    if not config.network.base_net_lock:
        data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    else:
        data_shape_dict = dict(train_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    if not config.network.base_net_lock:
        label_names = [k[0] for k in train_data.provide_label_single]
    else:
        label_names = []

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    from core.module import MutableModule
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
                        max_label_shapes=[max_label_shape for _ in range(batch_size)], fixed_param_prefix=fixed_param_prefix)

    # decide training params
    # metric
    from core import callback, metric
    if not config.network.base_net_lock:
        rpn_eval_metric = metric.RPNAccMetric()
        rpn_cls_metric = metric.RPNLogLossMetric()
        rpn_bbox_metric = metric.RPNL1LossMetric()
    rpn_fg_metric = metric.RPNFGFraction(config)
    eval_metric = metric.RCNNAccMetric(config)
    eval_fg_metric = metric.RCNNFGAccuracy(config)
    cls_metric = metric.RCNNLogLossMetric(config)
    bbox_metric = metric.RCNNL1LossMetric(config)
    eval_metrics = mx.metric.CompositeEvalMetric()

    if not config.network.base_net_lock:
        all_child_metrics = [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, rpn_fg_metric, eval_fg_metric, eval_metric, cls_metric, bbox_metric]
    else:
        all_child_metrics = [rpn_fg_metric, eval_fg_metric, eval_metric, cls_metric, bbox_metric]

    ################################################
    ### added / updated by Leonid to support oneshot
    ################################################
    if config.network.EMBEDDING_DIM != 0:
        all_child_metrics += [metric.RepresentativesMetric(config, None)]
        if config.network.EMBED_LOSS_ENABLED:
            all_child_metrics += [metric.EmbedMetric(config)]
        if config.network.REPS_CLS_LOSS:
            all_child_metrics += [metric.RepsCLSMetric(config)]
        if config.network.ADDITIONAL_LINEAR_CLS_LOSS:
            all_child_metrics += [metric.RCNNLinLogLossMetric(config)]
        if config.network.VAL_FILTER_REGRESS:
            all_child_metrics += [metric.ValRegMetric(config)]
    ################################################

    for child_metric in all_child_metrics:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = None #callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    epoch_end_callback = [] #[mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix, means, stds)]
    # decide learning rate
    # base_lr = lr
    # lr_factor = config.TRAIN.lr_factor
    # lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    # lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    # lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    # lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    # # print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    # from utils.lr_scheduler import WarmupMultiFactorScheduler
    # lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    lr_scheduler = None
    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'clip_gradient': None}
    #
    from utils.PrefetchingIter import PrefetchingIter
    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch, config=config)

    arg_params_, aux_params_ = mod.get_params()

    arg_params_['bbox_pred_weight'] = (arg_params_['bbox_pred_weight'].T * mx.nd.array(stds)).T
    arg_params_['bbox_pred_bias'] = arg_params_['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)

    return arg_params_, aux_params_

def gen_predictor(sym, arg_params, aux_params, data):
    label_names = []

    provide_label = [None for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]

    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(args.gpu)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor

# =================================================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='one-shot detection test')
    # general
    parser.add_argument('--test_name',      default='',     type=str, required=True, help='experiment name. Determines the dataset nad the model')
    parser.add_argument('--gpu',            default=0,      type=int, required=False, help='gpu number (ordinal) to use')
    parser.add_argument('--gen_episodes',   default=0,      type=int, required=False, help='set `1` to produce episodes for this task')
    parser.add_argument('--load_episodes',  default=1,      type=int, required=False, help='set `1` to load episodes from existing files')
    parser.add_argument('--resume',         default=0,      type=int, required=False, help='set `1` to load saved stats and continue from last episode');
    parser.add_argument('--do_finetune',    default=0,      type=int, required=False, help='set `1` to perform fine-tuning')
    parser.add_argument('--num_finetune_epochs',   default=10, type=int,    help='how many finetune epochs to run', required=False)
    parser.add_argument('--use_aug_reps',   default=0,      type=int, help='set `1` to use augmentation when generating representatives', required=False)
    parser.add_argument('--use_aug',        default=0,      type=int, help='set `1` to use augmentation for training', required=False)
    parser.add_argument('--num_aug_rounds', default=10,     type=int, help='how many augmentations to do', required=False)
    parser.add_argument('--exactNshot',     default=1,      type=int, help='ensure the number of training objects does not exceed Nshot', required=False)
    parser.add_argument('--ensure_quota',   default=0,        type=int, required=False, help='ensure the Nshot number of training objects, complementing from the pool if initial examples cannot be used')
    parser.add_argument('--Nshot',          default=1,      type=int, required=False, help='Number of samples per category')
    parser.add_argument('--Nway',           default=5,      type=int, required=False, help='Number of few-shot categories')
    parser.add_argument('--Nquery_cat',     default=10,     type=int, required=False, help='Number of query images per category')
    parser.add_argument('--Nepisodes',      default=500,    type=int, required=False, help='Number of episodes in benchmark')
    parser.add_argument('--Nreps_shot', default=2, type=int, required=False,      help='Number of representatives per category in episode')
    parser.add_argument('--score_thresh',   default=0,      type=float, required=False, help='Score threshold for computing the perf. statistics')
    parser.add_argument('--disp_score_thresh', default=0.05, type=float, required=False, help='Score threshold for display')

    parser.add_argument('--iou_thresh',     default=0.7,    type=float, required=False, help='IoU threshold for detecting training examples for representatives')
    parser.add_argument('--lr',             default=1e-4,   type=float, required=False, help='learning rate to use')
    parser.add_argument('--custom_exp',     default='',     type=str, required=False, help='additinal log filename suffix')
    parser.add_argument('--ovthresh',       default=0.5,    type=float, required=False, help='in perf. evaluation: consider GT bbox identified with detection bbox if the IoU is above this threshold')
    parser.add_argument('--scores_field',   default='cls_prob_reshape', type=str,help='Type of scores produced by the model. other option: cls_score')
    parser.add_argument('--topK',           default=0,      type=int, required=False, help='Produce only this number of (top-scored) detections from the image')
    parser.add_argument('--freeze',         default=0,      type=int, required=False, help='freeze base layers in fine-tuning. Use values 1, 2 (see according lists of layers in the code)')
    parser.add_argument('--start_episode',  default=0,      type=int, required=False, help='Episode number to start from (relevent when using a loaded set of episodes)')
    parser.add_argument('--isolated',       default=0,      type=int, required=False, help='performance evaluation for each episode in separate (not cumulative)')
    parser.add_argument('--n_rpn_props',    default=0,      type=int, required=False, help='Number of RPN region proposals per image (default=2000)')
    parser.add_argument('--display',        default=0,      type=int, required=False, help='Print out the detections')
    parser.add_argument('--nms_train',      default=0.3,      type=float, required=False, help='nms applied to relevant training image regions')
    parser.add_argument('--train_clustering', default=1, type=int, required=False,help='set `1` to produce the representatives via clustering of relevant training rois')
    parser.add_argument('--size_filter',    default=0, type=int, required=False, help='remove rois with less than size_filter portion of the image')
    parser.add_argument('--nqc',            default=0, type=int, required=False, help='test on just the firts nqc query images (relevant when working on loadd episodes)')
    parser.add_argument('--display_nImgs',  default=0, type=int, required=False, help='print the image indices for classes used in the benchmark')
    parser.add_argument('--validate',       default=0, type=int, required=False, help='set `1` to filter the input images according to validation lists')
    args = parser.parse_args()
    return args

args = parse_args()
nms_ovp_rois = gpu_nms_wrapper(args.nms_train, args.gpu)
nms_dets = gpu_nms_wrapper(config.TEST.NMS, args.gpu)


def run_detection    (sym, arg_params, aux_params, img_fname, img_cat, cat_indices,epi_cats_names,  exp_root, nImg,epi_num, score_thresh=args.score_thresh, scores_field=args.scores_field):
    data, data_batch, scale_factor,im = prep_data_single(img_fname)
    predictor = gen_predictor(sym, arg_params, aux_params, data)
    #nms = gpu_nms_wrapper(config.TEST.NMS, args.gpu)
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config, scores_field=scores_field)
    boxes = boxes[0].astype('f')
    scores = scores[0].astype('f')
    scores = scores[:, cat_indices]
    if not config.CLASS_AGNOSTIC:
        boxes = boxes[:,min(cat_indices)*4: (max(cat_indices)+1)*4]
    if args.topK>0:
        scores_seq = np.reshape(scores, (-1))
        scores_sorted = np.sort(-scores_seq)
        thresh = -scores_sorted[args.topK - 1]
        scores = (scores >= thresh) * scores

    dets_nms = []
    for j in range(scores.shape[1]):
        cls_scores = scores[:, j, np.newaxis]
        cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        keep = nms_dets(cls_dets)
        cls_dets = cls_dets[keep, :]
        cls_dets = cls_dets[cls_dets[:, -1] > score_thresh, :]
        dets_nms.append(cls_dets)

    # for det in dets_nms:
    #     print(det.shape)

    if args.display==1:
        save_fname = os.path.join(exp_root, 'epi_{0}_query_img:{1}_cat:{2}.png'.format(epi_num, nImg,img_cat))
        im = cv2.imread(img_fname)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        dets_disp = [[] for _ in dets_nms]
        for cls_idx, class_dets in enumerate(dets_nms):
            dets_disp[cls_idx] = class_dets[np.where(class_dets[:,4]>args.disp_score_thresh)]
        show_boxes(im, dets_disp, epi_cats_names, save_file_path=save_fname)

    return dets_nms

def get_workpoint():
    prep_reps_for_model = True # Computer representatives for the model, unless specified otherwise
    new_cats_to_beginning = False # replace the old classes, rather than ignoring them
    train_objects_fname = ''
    scores_fname = ''

    if args.test_name == 'RepMet_inloc':  # RepMet detector
        cfg_fname = root+'/experiments/cfgs/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8.yaml'
        test_classes_fname =root+'/data/Imagenet_LOC/in_domain_categories.txt'  # 214 categories
        roidb_fname = root+'/data/Imagenet_LOC/voc_inloc_roidb.pkl'
        model_case = args.test_name
        test_model_name = args.test_name
        # train_objects_fname, scores_fname = get_train_objects_fname('repmet_inloc')

    if args.test_name == 'Vanilla_inloc':  # 'nb19_214_train_hist_11':
        cfg_fname = root+'/experiments/cfgs/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19_noemb.yaml'
        test_classes_fname = root+'/data/Imagenet_LOC/in_domain_categories.txt'  # 214 categories
        roidb_fname = root + '/data/Imagenet_LOC/voc_inloc_roidb.pkl'
        model_case = args.test_name
        test_model_name = args.test_name
        #train_objects_fname, scores_fname = get_train_objects_fname('vanilla_inloc')

    return cfg_fname, model_case, test_classes_fname, test_model_name, roidb_fname,\
           train_objects_fname, scores_fname,prep_reps_for_model,new_cats_to_beginning

def get_datagen():
    aug_data_generator = None
    if args.use_aug_reps == 1 or args.use_aug == 1:
        from keras.preprocessing.image import ImageDataGenerator

        aug_data_generator = ImageDataGenerator(
            rotation_range=30,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )

    if args.use_aug_reps == 2 or args.use_aug == 2:
        from keras.preprocessing.image import ImageDataGenerator

        aug_data_generator = ImageDataGenerator(
            rotation_range=0,
            horizontal_flip=True,
            width_shift_range=0,
            height_shift_range=0,
            zoom_range=0
        )

    if args.use_aug_reps == 3 or args.use_aug == 3:
        from keras.preprocessing.image import ImageDataGenerator

        aug_data_generator = ImageDataGenerator(
            rotation_range=15,
            horizontal_flip=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05
        )
    return aug_data_generator

def get_names(test_model_name):
    exp_name = args.test_name+'_{0}shot_{1}way_{2}qpc_{3}epi'
    gen_root = root+'/output/benchmarks'
    exp_name_str = exp_name.format(args.Nshot, args.Nway, args.Nquery_cat, args.Nepisodes)
    episodes_fname = os.path.join(gen_root, exp_name_str + '_episodes.npz')

    run_name = exp_name_str
    if args.do_finetune==1:
        run_name +='_ft:{0}'.format(args.num_finetune_epochs)
    if args.topK>0:
        run_name += '_top'+str(args.topK)
    exp_root = assert_folder(os.path.join(gen_root, test_model_name, run_name))

    log_name = run_name+'_aug_{0}:{1}_lr_{2:.1e}_nms:{3:.2f}_iou:{4:.2f}_clust:{5}_exact:{6}'.format(args.use_aug,args.num_aug_rounds,args.lr,args.nms_train,args.iou_thresh,args.train_clustering,args.exactNshot)
    script_name = os.path.basename(__file__)
    exp_data_fname = os.path.join(exp_root, script_name[0:-3] + '_' + log_name + '_stats.npz')
    if args.custom_exp == '':
        log_filename = os.path.join(exp_root, script_name[0:-3] + '_'+log_name+ '_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M')))
    else:
        log_filename = os.path.join(exp_root, script_name[0:-3] + '_' + log_name +'_'+args.custom_exp+ '_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M')))
    return exp_root,log_filename,episodes_fname,exp_data_fname

def is_large_roi(img_h, img_w, bbox, min_portion):
    bbox = roidb[nImg]['boxes'][i_obj]
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    return 100 * max(height / roidb[nImg]['height'], width / roidb[nImg]['width']) < args.size_filter

def gen_reps(roidb,N_inst,epi_cats,train_nImg,model_case,sym,epi_root,epi_num,epi_cats_names,new_cats_to_beginning):
    # produce training features and add to model -------------------------------------------------------------------------------------------------------
    all_custom_embed = np.zeros((config.network.EMBEDDING_DIM, 0))
    arg_params, aux_params = load_a_model(config, model_case)
    all_cat_ids_heap = []
    all_img_indices = []
    for aug_round in range(N_inst):
        shot_cntr = np.zeros(args.Nway)
        for i_cat, cat in enumerate(epi_cats):
            for nImg in train_nImg[cat]:
                cur_img = roidb[nImg]['image']
                cur_bbs = roidb[nImg]['boxes']
                cats_img = np.expand_dims(roidb[nImg]['gt_classes'], axis=1)
                img_train_bbs = np.zeros((0, 5))
                if args.do_finetune == 1 and args.use_aug_reps > 0 and aug_round > 0:
                    # cur_img converts from image to path
                    cur_img = cv2.imread(cur_img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    cur_img_, cur_bbs_, cats_img_ = random_transform(
                        cur_img.copy(),
                        cur_bbs.copy(),
                        aug_data_generator,
                        cats_img=cats_img.copy(),
                    )
                    cur_img = cur_img_
                    cur_bbs = cur_bbs_
                    cats_img = cats_img_
                    if False: # save the image for debug purposes
                        save_fname = os.path.join(base_path, 'img_aug_' + str(iImg) + ',' + str(aug_round) + '.png')
                        tmp = show_boxes(cur_img, [cur_bbs], ['tmp'], save_file_path=save_fname)

                data, data_batch, scale_factor, im = prep_data_single(cur_img)
                predictor = gen_predictor(sym, arg_params, aux_params, data)
                out_data = predictor.predict(data_batch)
                rois_embed = out_data[0]['rois_output'].asnumpy()  # rois_boxes. shape: [#rois, 5]
                psp_embed = out_data[0]['psp_final_embed_output'].asnumpy()  # rois_feats. [#rois, #dims]

                train_bbs = np.float32(cur_bbs) * scale_factor
                train_bbs = np.concatenate((train_bbs, cats_img), axis=1)
                bb_indices, bb_cats = get_ovp_rois(train_bbs, rois_embed, np.array(args.iou_thresh), args.nms_train > 0)
                if len(bb_indices)==0: # no ROIs were found
                    if args.ensure_quota==1:
                        train_objects = {}
                        for e_cat in epi_cats:
                            train_objects[e_cat] = []
                        # fetch more images from this category
                        train_nImg, train_objects = get_cat_nImg(train_nImg,train_objects, cat,epi_cats)
                    continue

                # select those training objects that are within episode categories
                bb_indices_epi, cat_ids_epi = [[] for _ in range(2)]
                for bb_idx, bb_cat in zip(bb_indices, bb_cats):
                    if bb_cat in epi_cats:
                        cat_idx = np.where(epi_cats == bb_cat)[0][0]

                        # if args.size_filter>0:
                        #     bbox = rois_embed[bb_idx][1:]
                        #     height = (bbox[3] - bbox[1])/scale_factor
                        #     width = (bbox[2] - bbox[0])/scale_factor
                        #     if 100*max( height/roidb[nImg]['height'], width/roidb[nImg]['width'])<args.size_filter:
                        #         continue

                        if args.exactNshot == 1 and shot_cntr[cat_idx] >= args.Nshot * N_inst:
                            continue
                        shot_cntr[cat_idx] += 1
                        bb_indices_epi += [bb_idx]  # bounding box index
                        cat_ids_epi += [int(cat_idx)]  # index of the episode category for this bounding box
                # extract embedded ROIs per nImg ----------------------
                custom_embed = extract_embeds(psp_embed, bb_indices_epi)
                all_custom_embed = np.concatenate([all_custom_embed, custom_embed], axis=1)
                all_cat_ids_heap += cat_ids_epi
                if len(bb_indices_epi) == 0:
                    continue

                # print all bboxes in img ----------
                if args.display==1:
                    save_fname = os.path.join(epi_root, 'epi_{0}_trn_img:{1}_cat:{2}.png'.format(epi_num, nImg, epi_cats_names[cat_ids_epi[-1]]))
                    # im, dets_nms = get_disp_data(img_fname, scores, boxes, nms, thresh)
                    im = cv2.imread(cur_img)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    show_boxes_loc(im, rois_embed[np.asarray(bb_indices_epi), 1:], cat_ids_epi, scale=1 / scale_factor, save_file_path=save_fname)

            # print('got {0} embeddings, {1} objects '.format(all_custom_embed.shape[1],len(all_cat_ids_heap)))
    cat_ids_all = np.array(all_cat_ids_heap)
    tot_embeds = compute_tot_embeds(cat_ids_all, args.Nway, all_custom_embed, use_kmeans=args.train_clustering == 1)
    arg_params, new_reps, num_classes = add_reps_to_model(arg_params, tot_embeds,from_start=new_cats_to_beginning)

    config.dataset.NUM_CLASSES = num_classes
    n_ext_classes = config.dataset.NUM_CLASSES - 1

    return arg_params,n_ext_classes

def test_model(perf_stats,epi_cats,query_images,cat_indices,roidb,d,sym_ext, arg_params, aux_params,epi_root,epi_num,epi_cats_names, display=0):
    for cat in epi_cats:
        if args.nqc>0 and args.nqc< len(query_images[cat]):
            q_images = query_images[cat][0:args.nqc]#random.sample(query_images[cat],args.nqc)
        else:
            q_images = query_images[cat]
        for nImg in q_images:
            img_fname = roidb[nImg]['image']
            gt_classes = roidb[nImg]['gt_classes']
            if display==1:
                img_cats = [epi_cats_names[i] for i in range(args.Nway) if epi_cats[i] in gt_classes]
                q_dets = run_detection(sym_ext, arg_params, aux_params, img_fname, img_cats[0], cat_indices, epi_cats_names, epi_root, nImg, epi_num)
            else:
                q_dets = run_detection(sym_ext, arg_params, aux_params, img_fname, '', cat_indices, [], epi_root, nImg, epi_num)
            gt_boxes = np.copy(roidb[nImg]['boxes'])
            # legacy from Pascal dataset
            gt_boxes_test = []
            gt_classes_test = []
            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                if gt_class in epi_cats:
                    gt_boxes_test += [gt_box]
                    gt_classes_test += [gt_class]
            gt_classes_test = np.asarray(gt_classes_test)
            gt_boxes_test = np.asarray(gt_boxes_test)
            d = perf_stats.comp_epi_stats_m(d, q_dets, gt_boxes_test, gt_classes_test, epi_cats, args.ovthresh)
    return d

def get_cat_nImg(train_nImg,train_objects, cat,epi_cats):
    imgs_of_cat = cls2img[cat]
    nImg = random.sample(imgs_of_cat,1)[0]
    if nImg not in train_nImg[cat]:
        is_valid = True
        if args.validate==1:
            epi_scores = []
            idx = np.where(train_objects_ar[:, 0] == nImg)[0].tolist()
            if len(idx) > 0:
                if len(BG_scores[idx]) > 0:
                    epi_scores += BG_scores[idx].tolist()
            if len(epi_scores) > 0 and max(epi_scores) > sc_thresh:
                is_valid = False
                logger.info('Ignoring nImg {0} with score {1:.3f}'.format(nImg, max(epi_scores)))

        has_objects = False
        received_objects = roidb[nImg]['gt_classes'].tolist()
        for i_obj, rec_cat in enumerate(received_objects):
            if rec_cat not in epi_cats:
                continue
            if args.size_filter > 0:
                bbox = roidb[nImg]['boxes'][i_obj]
                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                if 100 * max(height / roidb[nImg]['height'], width / roidb[nImg]['width']) < args.size_filter:
                    continue
            has_objects = True
            train_objects[rec_cat] += [[nImg, i_obj]]

        if is_valid and has_objects:
            train_nImg[cat]+=[nImg]
    return train_nImg,train_objects

cfg_fname, model_case, test_classes_fname, test_model_name, roidb_fname,\
train_objects_fname, scores_fname,prep_reps_for_model,new_cats_to_beginning = get_workpoint()

# database ----------------------------------------------------------------------------------------
with open(roidb_fname, 'rb') as fid:
    roidb_data = cPickle.load(fid)
roidb = roidb_data['roidb']
cls2img = roidb_data['cls2img']
roidb_classes = roidb_data['classes']
test_cats, test_cats_names = names_list_to_indices(test_classes_fname, roidb_classes)

if args.validate == 1:
    with open(train_objects_fname, 'rb') as fid:
        train_data = cPickle.load(fid)
    train_objects = train_data['train_objects']
    train_objects_ar = np.asarray(train_objects)
    with open(scores_fname, 'rb') as fid:
        BG_scores = cPickle.load(fid)
    sc_thresh = 800


# workpoint --------------------------------------------
aug_data_generator = get_datagen()
update_config(cfg_fname)
exp_root,log_filename,episodes_fname,exp_data_fname = get_names(test_model_name)
logger = configure_logging(log_filename)
logger.info(' ********* test {0} started. *********'.format(args.test_name))
logger.info('Argument List:{}'.format(str(sys.argv)))

args_dict = args.__dict__
for arg_key in args_dict.keys():
    logger.info('{0}: {1}'.format(arg_key, args_dict[arg_key]))
def main(): # test_config


    # technical preparations ------------------------------------------------------------------------------------------------------
    MaxBboxesPerImage = 2000
    nms_factor = 0.1
    nd = np.int(args.Nepisodes * args.Nway * MaxBboxesPerImage * nms_factor)
    perf_stats_base = PerfStats(difficult=False, Nslots=nd)
    d_base = 0
    perf_stats = PerfStats(difficult=False, Nslots=nd)
    d = 0
    mxnwarmup = mx.nd.ones((1, 1), mx.gpu(args.gpu))

    N_inst = args.num_aug_rounds +1 if args.use_aug_reps else 1
    start_episode =args.start_episode
    end_episode =args.Nepisodes
    if args.load_episodes == 1:
        if os.path.exists(episodes_fname):
            with open(episodes_fname,'rb') as fid:
                episodes = cPickle.load(fid)
            end_episode = len(episodes)
        else:
            args.gen_episodes = 1
            args.load_episodes = 0

    if args.gen_episodes == 1:
        episodes = [{} for _ in range(args.Nepisodes)]

    if args.resume==1 and os.path.exists(exp_data_fname):
        stats_data = np.load(exp_data_fname)
        if stats_data['n_episode']>=end_episode-1:
            logger.info('resume halt: all episodes were already executed.')
            return
        perf_stats.merge_stats_ext(stats_data['stats'])
        d =perf_stats.d
        start_episode = stats_data['n_episode']+1
        perf_stats.print_perf(logger, prefix=args.test_name + ' resuming from epi {0}: '.format(start_episode))
    predictor = None



# main loop==================================================================================================================

    for epi_num in range(start_episode,end_episode):
        if args.isolated==1:
            d = 0
            d_base = 0
            perf_stats_base = PerfStats(difficult=False, Nslots=nd)
            perf_stats = PerfStats(difficult=False, Nslots=nd)

        if args.gen_episodes == 0:
            update_config(cfg_fname)
            if args.n_rpn_props>0:
                config.TEST.RPN_POST_NMS_TOP_N = args.n_rpn_props

            if config.network.REPS_PER_CLASS < args.Nreps_shot*args.Nshot:
                config.network.REPS_PER_CLASS = args.Nreps_shot*args.Nshot
            n_base_classes = config.dataset.NUM_CLASSES-1
            sym_instance = eval(config.symbol + '.' + config.symbol)()
            sym = sym_instance.get_symbol(config, is_train=False)
            arg_params, aux_params = load_a_model(config, model_case)


            config.TRAIN.UPDATE_REPS_VIA_CLUSTERING = False
            config.network.base_net_lock = False
            config.network.ADDITIONAL_LINEAR_CLS_LOSS = False
            #config.TRAIN.lr = 0.0001
            if args.freeze == 1:
                config.network.FIXED_PARAMS.extend(['res1', 'res2', 'res3', 'res4', 'res5', 'offset_', 'fpn_', 'rpn_conv', 'rpn_', 'fc_new_','bbox_pred'])
            if args.freeze == 2:
                config.network.FIXED_PARAMS.extend(['res1', 'res2', 'res3', 'res4', 'res5', 'offset_', 'fpn_', 'rpn_conv', 'rpn_'])

        if args.load_episodes == 1:
            epi_cats = episodes[epi_num]['epi_cats']
            train_nImg = episodes[epi_num]['train_nImg']
            query_images = episodes[epi_num]['query_images']
            epi_cats_names = episodes[epi_num]['epi_cats_names'] if args.display==1 else []
            if args.display_nImgs==1:
                cat_imgs = {}
                for cat in epi_cats:
                    cat_imgs[cat] = []
                for cat in epi_cats:
                    for nImg in train_nImg[cat]:
                        received_objects = roidb[nImg]['gt_classes'].tolist()
                        for rec_cat in received_objects:
                            cat_imgs[rec_cat]+=[nImg]
                for i_cat, cat in enumerate(epi_cats):
                    print('cat {0}------------------'.format(epi_cats_names[i_cat]))
                    print(cat_imgs[cat])

        else:  # -select images/objects to fill the request of #objects per category------------------------------------------------------------------------------------------------
            epi_data = random.sample(zip(test_cats,test_cats_names), args.Nway)
            epi_cats=[]
            epi_cats_names=[]
            for epi_entry in epi_data:
                epi_cats+=[epi_entry[0]]
                epi_cats_names+=[epi_entry[1]]

            train_nImg = {}
            train_objects = {}
            query_images = {}
            for cat in epi_cats:
                train_nImg[cat]=[]
                train_objects[cat]=[]
                query_images[cat] = []

            for cat in epi_cats:
                imgs_of_cat = cls2img[cat]
                if len(imgs_of_cat)==0:
                    sys.exit('empty list of images for the category {0}'.format(cat))
                while len(train_objects[cat])<args.Nshot:
                    train_nImg,train_objects = get_cat_nImg(train_nImg,train_objects,cat,epi_cats) # one attempt

                while len(query_images[cat])<args.Nquery_cat:
                    while True:
                        nImg = random.sample(imgs_of_cat,1)[0]
                        if nImg not in train_nImg[cat] and nImg not in query_images[cat]:
                            query_images[cat]+=[nImg]
                            break
            if args.gen_episodes==1:
                episodes[epi_num]['epi_cats'] = epi_cats
                episodes[epi_num]['train_nImg'] = train_nImg
                episodes[epi_num]['query_images'] = query_images
                episodes[epi_num]['epi_cats_names'] = epi_cats_names


        if args.display==1:
            epi_root = assert_folder(os.path.join(exp_root, 'epi_' + str(epi_num)))
            for cat, cat_name in zip(epi_cats,epi_cats_names):
                with open(os.path.join(epi_root,'cat_'+cat_name),'w') as fid:
                    for nImg in train_nImg[cat]:
                        fid.write('%d\n' % nImg)
        else:
            epi_root=''

        if args.gen_episodes == 1:
            continue

        arg_params, aux_params = load_a_model(config, model_case)

# =============================================================================================================
        if prep_reps_for_model:
            arg_params, n_ext_classes = gen_reps(roidb,N_inst,epi_cats,train_nImg,model_case,\
                                                 sym,epi_root,epi_num,epi_cats_names,new_cats_to_beginning)

        sym_ext = sym_instance.get_symbol(config, is_train=False)
        if new_cats_to_beginning:#rgs.test_name=='base_inloc':
            n_base_classes=0
            n_ext_classes = args.Nway
        ap_before = -1

        # detection before finetuning ----------------------------------------------------------
        if args.do_finetune==1:
            cat_indices = range(n_base_classes + 1, n_ext_classes + 1)
            d_base = test_model(perf_stats_base,epi_cats, query_images, cat_indices, roidb, d_base,sym_ext, arg_params, aux_params,epi_root,epi_num,epi_cats_names)
            perf_stats_base.print_perf(logger, prefix=args.test_name + ' no ft: epi {0} {1}-shot {2}-way: '.format(epi_num, args.Nshot, args.Nway))
            tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap = perf_stats_base.compute_stats(0)
            ap_before = ap

        # ---------------------------------------------------------------------------------------------------------
        # fine tune on the current train data
        shot_cntr = np.zeros(args.Nway)
        if args.do_finetune==1:
            print('fine-tuning...')
            if args.use_aug>0:
                aug_gen = lambda img, boxes: random_transform(img.copy(), boxes[:, :4].copy(), aug_data_generator)
            else:
                aug_gen = None
            n_roidb = []

            for cat in epi_cats:
                for nImg in train_nImg[cat]:
                    cur_img = roidb[nImg]['image']
                    cur_bbs = roidb[nImg]['boxes']
                    cats_img = np.expand_dims(roidb[nImg]['gt_classes'], axis=1)
                    BBs = np.zeros((0, 5))
                    train_bbs = np.float32(cur_bbs)
                    train_bbs = np.concatenate((train_bbs, cats_img), axis=1)
                    for cat_img, train_bb in zip(cats_img, train_bbs):
                        if cat_img in epi_cats:
                            cat_idx = np.where(epi_cats == cat_img)[0][0]
                            train_bb[4] = cat_idx
                            BB = np.expand_dims(train_bb, axis=0)
                            BBs = np.concatenate((BBs, np.expand_dims(train_bb, axis=0)), axis=0)
                    if BBs.shape[0]==0:
                        continue

                    BBs = np.array(BBs).astype(np.uint16)

                    for BB in BBs:
                        cat_idx = BB[4]
                        if args.exactNshot == 1 and shot_cntr[cat_idx] >= args.Nshot*N_inst:
                            continue
                        shot_cntr[cat_idx]+=1
                        oneHot = np.zeros((n_ext_classes),dtype=np.float32)
                        oneHot[cat_idx+n_base_classes]=1
                        im = cv2.imread(roidb[nImg]['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        class_array = np.expand_dims(BB[4],axis=0)
                        if BB.shape[0]==0:
                            continue
                        n_roidb.append({
                            'boxes': np.expand_dims(BB[:4],axis=0),
                            'flipped': False,
                            'gt_classes': n_base_classes+class_array.astype(np.int32)+1,
                            'gt_overlaps': oneHot,
                            'height': im.shape[0],
                            'image': roidb[nImg]['image'],
                            'max_classes': n_base_classes+class_array,
                            'max_overlaps': np.ones((1,1),dtype=np.float32),
                            'width': im.shape[1],
                            'aug_gen': aug_gen
                        })
            print('roidb has {0} samples.'.format(len(n_roidb)))
            config.TRAIN.UPDATE_REPS_VIA_CLUSTERING = False
            if args.test_name == 'base_inloc':
                tot_n_classes = n_base_classes + n_ext_classes
                arg_params['cls_score_bias']=arg_params['cls_score_bias'][0:(tot_n_classes+1)]
                arg_params['cls_score_weight'] = arg_params['cls_score_weight'][0:(tot_n_classes + 1)]
                arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'][0:4*(tot_n_classes + 1)]
                arg_params['bbox_pred_weight'] = arg_params['bbox_pred_weight'][0:4*(tot_n_classes + 1)]

            if False:
                import copy
                arg_params_ = copy.deepcopy(arg_params)

            print('Starting training...')
            arg_params, aux_params = train_net(
                sym_instance,
                lr=args.lr,
                begin_epoch=0, end_epoch=args.num_finetune_epochs,
                ctx=[mx.gpu(args.gpu)], roidb=n_roidb,
                arg_params=arg_params, aux_params=aux_params
            )

            sym_ext = sym_instance.get_symbol(config, is_train=False)

        # detection ---------------------------------------------------------------------------------------------------------
        if new_cats_to_beginning:
            cat_indices = range(1, args.Nway + 1)
        else:
            cat_indices = range(n_base_classes + 1, n_ext_classes + 1)
        d = test_model(perf_stats,epi_cats, query_images, cat_indices, roidb,d,sym_ext, arg_params, aux_params,epi_root,epi_num,epi_cats_names,args.display)

        perf_stats.print_perf(logger, prefix=args.test_name + ' epi {0} {1}-shot {2}-way: '.format(epi_num, args.Nshot, args.Nway))
        if args.do_finetune == 1:
            tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap = perf_stats.compute_stats(0)
            ap_after = ap
            logger.info('AP change due to finetuning: {0:.3f}-->{1:.3f}'.format(ap_before,ap_after))

    stats = perf_stats.get_stats()  # [ [self.sc, self.tp, self.fp, self.fpw, self.fpb], self.nGT, self.d,self.img_recAtK]
    np.savez(exp_data_fname, stats=stats)

    if args.gen_episodes==1:
        with open(episodes_fname,'wb') as fid:
            cPickle.dump(episodes,fid,protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
