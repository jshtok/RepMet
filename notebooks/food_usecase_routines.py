import cv2
import os
import sys
import argparse
import copy
import numpy as np

sys.path.append('../fpn')
sys.path.append('../lib')
sys.path.append('../lib/utils')
sys.path.append('../lib/nms')

from miscellaneous import assert_folder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils.show_boxes import disp_dets
data_root = '../data/Food/food_usecase_data'
disp_folder = assert_folder('../output/food_usecase_data')
prd2ename_fname = '../data/Food/foods_prd2ename.csv'
prd2ename_data = np.load(prd2ename_fname)
prd2ename = prd2ename_data['prd2ename']

def get_dets_class_names(dets_n):
    for det in dets_n:
        class_names = []
        for prd in det[3]:
            if prd in prd2ename.keys():
                class_names += [prd2ename[prd]]
            else:
                class_names += [prd]
        det += [class_names]
    return dets_n

# ------------------------------------------------------------------------------------------------------------------------

def print_typical_tray_multiview():
    # imgname_left = os.path.join(data_root, '1007_10000000006530_left.jpg')
    # imgname_top = os.path.join(data_root, '1007_10000000006530_top.jpg')
    # imgname_right = os.path.join(data_root, '1007_10000000006530_right.jpg')
    imgname_left = os.path.join(data_root, '15849_left.jpg')
    imgname_top = os.path.join(data_root, '15849_top.jpg')
    imgname_right = os.path.join(data_root, '15849_right.jpg')
    img_left = mpimg.imread(imgname_left)
    img_top = mpimg.imread(imgname_top)
    img_right = mpimg.imread(imgname_right)

    fig = plt.figure(1)
    ff = 3
    fig.set_size_inches((ff*8.5,3*ff*11),forward=False)

    plt.subplot(131)
    plt.imshow(img_left)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img_top)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img_right)
    plt.axis('off')

def display_few_shot_examples():

    # image_set = ['PRDS990000000000000025_0_192_501_589_885_top.jpg','PRDS990000000000000024_0_119_137_523_447_top.jpg',
    #              'PRDS990000000000000023_0_118_208_470_612_top.jpg','PRDS990000000000000021_0_571_234_923_608_top.jpg']
    image_set = ['5cadb37d4b967f67d3047964.jpg','12686073-1-white.jpg','lacoste-logo-sweatshirt-white-p12654-72879_image.jpg']
    fig = plt.figure(2)
    ff = 2
    fig.set_size_inches((ff*8.5,3*ff*11),forward=False)

    for cnt,img_basename in enumerate(image_set):
        imgname = os.path.join(data_root,img_basename)
        img = mpimg.imread(imgname)
        plt.subplot(1,3,cnt+1)
        plt.imshow(img)
        plt.axis('off')

# def disp_dets(img,dets,save_file_path):
#
#     import matplotlib.pyplot as plt
#     fig = plt.figure(1)
#     ff = 2
#     fig.set_size_inches((ff*8.5,ff*11),forward=False)
#     plt.axis("off")
#     img_d = cv2.cvtColor(copy.deepcopy(img),cv2.COLOR_BGR2RGB)
#
#     # print just the boxes first: ------------
#     for det in dets:
#         bbox = det[0]
#         scores = det[1]
#         labels = det[3]
#         pn = det[4]
#
#         left = int(bbox[0])
#         top = int(bbox[1])
#         right =  int(bbox[2])
#         bottom = int(bbox[3])
#         if pn=='p':
#             cv2.rectangle(img_d, (left, top), (right, bottom)
#                           , (255, 0, 0), 4)
#         elif pn=='n':
#             cv2.rectangle(img_d, (left, top), (right, bottom)
#                           , (0,255, 0), 4)
#         elif pn=='c':
#             cv2.rectangle(img_d, (left, top), (right, bottom)
#                           , (0,0,255), 4)
#         elif pn=='a':
#             cv2.rectangle(img_d, (left, top), (right, bottom)
#                           , (0,255,255), 4)
#
#     # print the texts -----------------------
#     for det in dets:
#         bbox = det[0]
#         scores = det[1]
#         labels = det[3]
#         pn = det[4]
#
#         left = int(bbox[0])
#         top = int(bbox[1])
#
#         text = '{0} - {1:.3f}'.format(labels[0], scores[0])
#         string_len = len(text)*13
#         bk_w = 17
#         FontScale = 0.6
#         cv2.rectangle(img_d, (left-2, top - bk_w), (left + string_len, top + bk_w), (255, 255, 255), cv2.FILLED)
#
#         #cv2.putText(img_d, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, FontScale, color=(0, 0, 0), thickness=2)
#         cv2.putText(img_d, text, (left, top), cv2.FONT_HERSHEY_TRIPLEX, FontScale, color=(0, 0, 0), thickness=1)
#
#         if len (det)==6:
#             text = det[5][0]
#             string_len = len(text) * 13
#             cv2.rectangle(img_d, (left-2, top - bk_w+2*bk_w), (left + string_len, top + bk_w+2*bk_w), (255, 255, 255), cv2.FILLED)
#             #cv2.putText(img_d, text, (left, top+2*bk_w), cv2.FONT_HERSHEY_SIMPLEX, FontScale, color=(0, 0, 0), thickness=2)
#             cv2.putText(img_d, text, (left, top+2*bk_w), cv2.FONT_HERSHEY_TRIPLEX, FontScale, color=(0, 0, 0), thickness=1)
#
#     plt.imshow(img_d)
#     fig.savefig(save_file_path)

def test_on_query_image(fs_serv,test_img_fname,det_engines):
    img = cv2.imread(test_img_fname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    score_thresh = 0.1
    dets_n = fs_serv.detect_on_image(img, score_thresh=score_thresh, det_engines=det_engines)
    dets_n = get_dets_class_names(dets_n)

    test_img_basename = os.path.basename(test_img_fname)
    img_file_path = os.path.join(disp_folder, test_img_basename.replace('.jpg', '_disp_n.jpg'))

    disp_dets(img, dets_n, img_file_path,figure_factor=2.0)

def get_box_proposal(fs_serv,img_path):
    from show_boxes import show_detsB_boxes

    q_dets_p = fs_serv.get_box_proposal(img_path)
    image_basename = os.path.basename(img_path)
    save_file_path = os.path.join(disp_folder, 'box_prop_{0}'.format(image_basename))
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION),
                       cv2.COLOR_BGR2RGB)
    show_detsB_boxes(img, q_dets_p, save_file_path=save_file_path)