import cv2
import os
import sys
import argparse
import numpy as np

sys.path.append('../fpn')
sys.path.append('../lib')
sys.path.append('../lib/utils')
sys.path.append('../lib/nms')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from FSD_serv import FSD_serv,get_FSD_serv_args
from FSD_common_lib import gen_train_zip, assert_folder,export_dets_CSV_from_multi

from easydict import EasyDict as edict
args = edict()
