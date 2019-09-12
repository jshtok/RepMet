# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Guodong Zhang
# --------------------------------------------------------		  

import argparse
import pprint
import logging
import time
import os
import mxnet as mx

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, pred_eval
from utils.load_model import load_param


def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None,nms_dets=None,is_docker=False):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    #pprint.pprint(cfg)
    #logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    #leonid: added to support ; for multi-dataset listing - this is a temp solution allowing just one DB in test
    datasets = dataset.split(';')
    dataset_paths = dataset_path.split(';')
    imagesets = image_set.split(';')
    output_paths = output_path.split(';')
    categ_index_offs = 20 #TODO: remove
    for dataset, dataset_path, image_set,output_path in zip(datasets,dataset_paths,imagesets,output_paths):
        if len(image_set.strip())<=0:
            continue

        if 'classes_list_fname' not in cfg.dataset:
            classes_list_fname = ''
        else:
            classes_list_fname = cfg.dataset.classes_list_fname

        # load symbol and testing data
        if has_rpn:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol(cfg, is_train=False)
            imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path,classes_list_fname=classes_list_fname,categ_index_offs=categ_index_offs)
            roidb = imdb.gt_roidb()
        else:
            sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
            sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
            imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
            gt_roidb = imdb.gt_roidb()
            roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)
        categ_index_offs+=imdb.num_classes
        # get test data iter
        test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

        if not is_docker:
            # load model
            arg_params, aux_params = load_param(prefix, epoch, process=True)

            # infer shape
            data_shape_dict = dict(test_data.provide_data_single)
            sym_instance.infer_shape(data_shape_dict)

            sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

            # decide maximum shape
            data_names = [k[0] for k in test_data.provide_data_single]
            label_names = None
            max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
            if not has_rpn:
                max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

            # create predictor
            predictor = Predictor(sym, data_names, label_names,
                                  context=ctx, max_data_shapes=max_data_shape,
                                  provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                                  arg_params=arg_params, aux_params=aux_params)

            #make sure imdb and detector have the same number of classes
            #imdb.num_classes=min(imdb.num_classes,cfg.dataset.NUM_CLASSES) # JS, March 2019: the JES dataset class produces num_classes = number of foreground classes, while the tester assumes this includes the background.
            imdb.num_classes =cfg.dataset.NUM_CLASSES
        else:
            predictor=None

        # start detection
        pred_eval(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger,nms_dets=nms_dets)

