# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------		

import _init_paths

#import cv2
import argparse
import os
import sys
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--debug', default=0, help='experiment configure file name', required=False, type=int)
    parser.add_argument('--is_docker', help='test in docker mode', action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger
from nms.nms import gpu_nms_wrapper

def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args
    gpu_nums = [int(i) for i in config.gpus.split(',')]
    nms_dets = gpu_nms_wrapper(config.TEST.NMS, gpu_nums[0])
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    output_path = os.path.join(final_output_path, '..', '+'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix)
    test_rcnn(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, output_path, config.TEST.test_epoch, args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal,
              args.thresh, logger=logger, output_path=final_output_path, nms_dets=nms_dets, is_docker=args.is_docker)

if __name__ == '__main__':
    main()
