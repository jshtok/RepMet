# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------														  
import _init_paths

import cv2
import argparse
import pprint
import os
import sys
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--debug', default=0, help='experiment configure file name', required=False, type=int)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core.loader import PyramidAnchorIterator
from core import callback, metric
from core.module import MutableModule
from utils.create_logger import create_logger
from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler

def ListList2ndarray(LList):
    dim_in = len(LList)
    dim_out = len(LList[0])
    weight = mx.ndarray.zeros([dim_in, dim_out, ])
    for i, inp in enumerate(LList):
        weight[i, :] = mx.nd.array(inp)
    weight = weight.T
    return weight

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    mx.random.seed(3)
    np.random.seed(3)
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    config['final_output_path'] = final_output_path

    # load symbol
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=True)

    feat_pyramid_level = np.log2(config.network.RPN_FEAT_STRIDE).astype(int)
    feat_sym = [sym.get_internals()['rpn_cls_score_p' + str(x) + '_output'] for x in feat_pyramid_level]

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    #leonid: adding semicolumn ";" support to allow several different datasets to be merged
    datasets = config.dataset.dataset.split(';')
    image_sets = config.dataset.image_set.split(';')
    data_paths = config.dataset.dataset_path.split(';')
    if type(config.dataset.per_category_epoch_max) is str:
        per_category_epoch_max = [float(x) for x in config.dataset.per_category_epoch_max.split(';')]
    else:
        per_category_epoch_max = [float(config.dataset.per_category_epoch_max)]
    roidbs = []
    categ_index_offs = 0
    if 'classes_list_fname' not in config.dataset:
        classes_list_fname =''
    else:
        classes_list_fname = config.dataset.classes_list_fname

    if 'num_ex_per_class' not in config.dataset:
        num_ex_per_class=''
    else:
        num_ex_per_class = config.dataset.num_ex_per_class


    for iD, dataset in enumerate(datasets):
        # load dataset and prepare imdb for training
        image_sets_cur = [iset for iset in image_sets[iD].split('+')]
        for image_set in image_sets_cur:
            cur_roidb, cur_num_classes = load_gt_roidb(
                dataset,
                image_set,
                config.dataset.root_path,
                data_paths[iD],
                flip=config.TRAIN.FLIP,
                per_category_epoch_max=per_category_epoch_max[iD],
                return_num_classes=True,
                categ_index_offs=categ_index_offs,
                classes_list_fname=classes_list_fname,
                num_ex_per_class = num_ex_per_class)

            roidbs.append(cur_roidb)
        categ_index_offs+=cur_num_classes
        # roidbs.extend([
        #     load_gt_roidb(
        #         dataset,
        #         image_set,
        #         config.dataset.root_path,
        #         data_paths[iD],
        #         flip=config.TRAIN.FLIP,
        #         per_category_epoch_max=per_category_epoch_max[iD])
        #     for image_set in image_sets])
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, config)

    # load training data

    train_data = PyramidAnchorIterator(feat_sym, roidb, config, batch_size=input_batch_size, shuffle=config.TRAIN.SHUFFLE,
                                       ctx=ctx, feat_strides=config.network.RPN_FEAT_STRIDE, anchor_scales=config.network.ANCHOR_SCALES,
                                       anchor_ratios=config.network.ANCHOR_RATIOS, aspect_grouping=config.TRAIN.ASPECT_GROUPING,
                                       allowed_border=np.inf)

    # infer max shape
    max_data_shape = [('data', (config.TRAIN.BATCH_IMAGES, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]
    max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
    max_data_shape.append(('gt_boxes', (config.TRAIN.BATCH_IMAGES, 100, 5)))
    print 'providing maximum shape', max_data_shape, max_label_shape

    if not config.network.base_net_lock:
        data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    else:
        data_shape_dict = dict(train_data.provide_data_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)

    # load and initialize params
    if config.TRAIN.RESUME:
        print('continue training from ', begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        sym_instance.init_weight(config, arg_params, aux_params)

    if config.TRAIN.LOAD_EMBEDDING:
        import cPickle
        with open(config.TRAIN.EMBEDDING_FNAME, 'rb') as fid:
            model_data = cPickle.load(fid)
        for fcn in ['1', '2', '3']:
            layer = model_data['dense_' + fcn]
            weight = ListList2ndarray(layer[0])
            bias = mx.nd.array(layer[1])
            arg_params['embed_dense_' + fcn + '_weight'] = weight
            arg_params['embed_dense_' + fcn + '_bias'] = bias

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    alt_fixed_param_prefix = config.network.ALT_FIXED_PARAMS
    data_names = [k[0] for k in train_data.provide_data_single]
    if not config.network.base_net_lock:
        label_names = [k[0] for k in train_data.provide_label_single]
    else:
        label_names = []

    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, max_data_shapes=[max_data_shape for _ in range(batch_size)],
                        max_label_shapes=[max_label_shape for _ in range(batch_size)], fixed_param_prefix=fixed_param_prefix,
                        alt_fixed_param_prefix=alt_fixed_param_prefix)

    # Leonid: Comment out the following two lines if switching to smaller number of GPUs and resuming training, then after it starts running un-comment back
    # if config.TRAIN.RESUME:
    #     mod._preload_opt_states = '%s-%04d.states'%(prefix, begin_epoch)
    #TODO: release this.
    # decide training params
    # metric
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

    # rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric
    if not config.network.base_net_lock:
        all_child_metrics = [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, rpn_fg_metric, eval_fg_metric, eval_metric, cls_metric, bbox_metric]
    else:
        all_child_metrics = [rpn_fg_metric, eval_fg_metric, eval_metric, cls_metric, bbox_metric]
    # all_child_metrics = [rpn_eval_metric, rpn_bbox_metric, rpn_fg_metric, eval_fg_metric, eval_metric, cls_metric, bbox_metric]

    ################################################
    ### added / updated by Leonid to support oneshot
    ################################################
    if config.network.EMBEDDING_DIM != 0:
        if config.network.EMBED_LOSS_ENABLED:
            all_child_metrics += [metric.RepresentativesMetric(config, final_output_path)] # moved from above. JS.
            all_child_metrics += [metric.EmbedMetric(config)]
            if config.network.BG_REPS:
                all_child_metrics += [metric.BGModelMetric(config)]
        if config.network.REPS_CLS_LOSS:
            all_child_metrics += [metric.RepsCLSMetric(config)]
        if config.network.ADDITIONAL_LINEAR_CLS_LOSS:
            all_child_metrics += [metric.RCNNLinLogLossMetric(config)]
        if config.network.VAL_FILTER_REGRESS:
            all_child_metrics += [metric.ValRegMetric(config)]
        if config.network.SCORE_HIST_REGRESS:
            all_child_metrics += [metric.ScoreHistMetric(config)]
    ################################################

    for child_metric in all_child_metrics:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    means = np.tile(np.array(config.TRAIN.BBOX_MEANS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    stds = np.tile(np.array(config.TRAIN.BBOX_STDS), 2 if config.CLASS_AGNOSTIC else config.dataset.NUM_CLASSES)
    epoch_end_callback = [mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True), callback.do_checkpoint(prefix, means, stds)]
    # decide learning rate
    base_lr = lr
    lr_factor = config.TRAIN.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
#    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)
    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'clip_gradient': None}
    #
    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    if args.debug==1:
        import copy
        arg_params_ = copy.deepcopy(arg_params)
        aux_params_ = copy.deepcopy(aux_params)

    # train
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch, config=config)

    if args.debug == 1:
        t = dictCompare(aux_params_, aux_params)
        t = dictCompare(arg_params_, arg_params)

def dictCompare( d1, d2 ):
    diff_items = {k: np.max(np.abs(d1[k].asnumpy()-d2[k].asnumpy())) for k in d1 if (k in d2) and np.array_equal(d1[k].asnumpy().shape,d2[k].asnumpy().shape) and not np.array_equal(d1[k].asnumpy(),d2[k].asnumpy())}
    diff_items_extra1 = {k: d1[k] for k in d1 if k not in d2}
    diff_items_extra2 = {k: d2[k] for k in d2 if k not in d1}
    diff_items_extra3 = {k: (d1[k].asnumpy().shape,d2[k].asnumpy().shape) for k in d1 if (k in d2) and not np.array_equal(d1[k].asnumpy().shape, d2[k].asnumpy().shape)}
    print(diff_items)
    print(diff_items_extra1.keys())
    print(diff_items_extra2.keys())
    print(diff_items_extra3)
    return diff_items, diff_items_extra1, diff_items_extra2, diff_items_extra3

def main():
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)

if __name__ == '__main__':
    main()
