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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


data_root = '../data/Food/food_usecase_data'
disp_folder = '../output/food_usecase_data'
prd2ename_fname = '../data/Food/foods_prd2ename.csv'
prd2ename_data = np.load(prd2ename_fname)
prd2ename = prd2ename_data['prd2ename']

def display_few_shot_examples():
    data_root = '/dccstor/jsdata1/dev/RepMet/notebooks/food_usecase_data'
    image_set = ['PRDS990000000000000025_0_192_501_589_885_top.jpg','PRDS990000000000000024_0_119_137_523_447_top.jpg',
                 'PRDS990000000000000023_0_118_208_470_612_top.jpg','PRDS990000000000000021_0_571_234_923_608_top.jpg']
    # nrows = int(np.sqrt(len(image_set)))
    # ncols = np.ceil(len(image_set)/nrows)
    nrows = 1
    ncols = 4
    #print('nrows={0},ncols={1}'.format(nrows, ncols))
    fig = plt.figure(2)
    ff = 2
    fig.set_size_inches((ff*8.5,3*ff*11),forward=False)

    for cnt,img_basename in enumerate(image_set):
        imgname = os.path.join(data_root,img_basename)
        img = mpimg.imread(imgname)
        plt.subplot(nrows,ncols,cnt+1)
        plt.imshow(img)
        plt.axis('off')


def print_typical_tray_multiview():
  imgname_left = os.path.join(data_root, '1007_10000000006530_left.jpg')
  imgname_top = os.path.join(data_root, '1007_10000000006530_top.jpg')
  imgname_right = os.path.join(data_root, '1007_10000000006530_right.jpg')
  img_left = mpimg.imread(imgname_left)
  img_top = mpimg.imread(imgname_top)
  img_right = mpimg.imread(imgname_right)

  fig = plt.figure(1)
  ff = 3
  fig.set_size_inches((ff * 8.5, 3 * ff * 11), forward=False)

  plt.subplot(131)
  plt.imshow(img_left)
  plt.axis('off')
  plt.subplot(132)
  plt.imshow(img_top)
  plt.axis('off')
  plt.subplot(133)
  plt.imshow(img_right)
  plt.axis('off')


# def print_typical_tray_multiview():
#     data_root = '/dccstor/jsdata1/dev/RepMet/notebooks/food_usecase_data'
#     imgname_left = os.path.join(data_root, '1007_10000000006530_left.jpg')
#     imgname_top = os.path.join(data_root, '1007_10000000006530_top.jpg')
#     imgname_right = os.path.join(data_root, '1007_10000000006530_right.jpg')
#     img_left = mpimg.imread(imgname_left)
#     img_top = mpimg.imread(imgname_top)
#     img_right = mpimg.imread(imgname_right)
#
#     fig = plt.figure(1)
#     ff = 3
#     fig.set_size_inches((ff*8.5,3*ff*11),forward=False)
#
#     plt.subplot(131)
#     plt.imshow(img_left)
#     plt.axis('off')
#     plt.subplot(132)
#     plt.imshow(img_top)
#     plt.axis('off')
#     plt.subplot(133)
#     plt.imshow(img_right)
#     plt.axis('off')


def disp_dets(img,dets,save_file_path):
    # display top-1 detection from each multi-candidate input
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    ff = 2
    fig.set_size_inches((ff*8.5,ff*11),forward=False)
    plt.axis("off")
    img_d = cv2.cvtColor(copy.deepcopy(img),cv2.COLOR_BGR2RGB)


    # print just the boxes first: ------------
    for det in dets:
        bbox = det[0]
        scores = det[1]
        labels = det[3]
        pn = det[4]

        left = int(bbox[0])
        top = int(bbox[1])
        right =  int(bbox[2])
        bottom = int(bbox[3])
        if pn=='p':
            cv2.rectangle(img_d, (left, top), (right, bottom)
                          , (255, 0, 0), 4)
        elif pn=='n':
            cv2.rectangle(img_d, (left, top), (right, bottom)
                          , (0,255, 0), 4)
        elif pn=='c':
            cv2.rectangle(img_d, (left, top), (right, bottom)
                          , (0,0,255), 4)
        elif pn=='a':
            cv2.rectangle(img_d, (left, top), (right, bottom)
                          , (0,255,255), 4)

    # print the texts -----------------------
    for det in dets:
        bbox = det[0]
        scores = det[1]
        labels = det[3]
        pn = det[4]

        left = int(bbox[0])
        top = int(bbox[1])

        text = '{0} - {1:.3f}'.format(labels[0], scores[0])
        string_len = len(text)*13
        bk_w = 17
        FontScale = 0.6
        cv2.rectangle(img_d, (left-2, top - bk_w), (left + string_len, top + bk_w), (255, 255, 255), cv2.FILLED)

        #cv2.putText(img_d, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, FontScale, color=(0, 0, 0), thickness=2)
        cv2.putText(img_d, text, (left, top), cv2.FONT_HERSHEY_TRIPLEX, FontScale, color=(0, 0, 0), thickness=1)

        if len (det)==6:
            text = det[5][0]
            string_len = len(text) * 13
            cv2.rectangle(img_d, (left-2, top - bk_w+2*bk_w), (left + string_len, top + bk_w+2*bk_w), (255, 255, 255), cv2.FILLED)
            #cv2.putText(img_d, text, (left, top+2*bk_w), cv2.FONT_HERSHEY_SIMPLEX, FontScale, color=(0, 0, 0), thickness=2)
            cv2.putText(img_d, text, (left, top+2*bk_w), cv2.FONT_HERSHEY_TRIPLEX, FontScale, color=(0, 0, 0), thickness=1)

    plt.imshow(img_d)
    fig.savefig(save_file_path)

def test_on_query_image(fs_serv,test_img_fname,det_engines):
    prd2ename_fname = '/dccstor/jsdata1/dev/RepMet/data/JES_pilot/all_GT.csv_converted_Feb24_prd2ename.csv'
    prd2ename_data = np.load(prd2ename_fname)
    prd2ename = prd2ename_data['prd2ename']

    img = cv2.imread(test_img_fname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    disp_folder = '../notebooks/food_usecase_data'

    #det_engines = 1
    score_thresh = 0.1
    dets_n = fs_serv.detect_on_image(img, score_thresh=score_thresh, num_candidates=3, det_engines=det_engines)

    for det in dets_n:
        class_names = []
        for prd in det[3]:
            if prd in prd2ename.keys():
                class_names += [prd2ename[prd]]
            else:
                class_names += [prd]
        det += [class_names]

    test_img_basename = os.path.basename(test_img_fname)
    img_file_path = os.path.join(disp_folder, test_img_basename.replace('.jpg', '_disp_n.jpg'))
    det_file_path = os.path.join(disp_folder, test_img_basename.replace('.jpg', '_dets_n.txt'))

    disp_dets(img, dets_n, img_file_path)

def get_box_proposal(fs_serv,img_path):
    from show_boxes import show_detsB_boxes

    output_folder = '../notebooks/food_usecase_data'
    q_dets_p = fs_serv.get_box_proposal(img_path)
    image_basename = os.path.basename(img_path)
    save_file_path = os.path.join(output_folder, 'box_prop_{0}'.format(image_basename))
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION),
                       cv2.COLOR_BGR2RGB)
    show_detsB_boxes(img, q_dets_p, save_file_path=save_file_path)
