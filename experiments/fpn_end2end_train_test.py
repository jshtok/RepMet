# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Guodong Zhang
# --------------------------------------------------------
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

# os.environ['MXNET_GPU_WORKER_NTHREADS'] = '4'
# os.environ['MXNET_GPU_COPY_NTHREADS'] = '4'
# os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'fpn'))

import matplotlib
matplotlib.use('agg')

import train_end2end
import test

# leonid: MXNet warmup
import mxnet as mx
#mxnetwarmup = mx.nd.ones((1,1), mx.gpu(0))

if __name__ == "__main__":
    train_end2end.main()
    test.main()




