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

from utils.show_boxes import disp_dets2
data_root = '../data/Logo/logo_usecase_data'
disp_folder = assert_folder('../output/logo_usecase_data')

def disp_dets2(img,dets,save_file_path):
    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    fig.set_size_inches((1.4*8.5,1.4*11),forward=False)
    plt.axis("off")
    img_d = cv2.cvtColor(copy.deepcopy(img),cv2.COLOR_BGR2RGB)
    w = 12
    for det in dets:
        bbox = det[0]
        scores = det[1]
        labels = det[3]
        pn = det[4]

        left = int(bbox[0])
        top = int(bbox[1])
        # right = left + int(bbox[2])
        # bottom = top + int(bbox[3])
        right =  int(bbox[2])
        bottom = int(bbox[3])
        if pn=='p':
            cv2.rectangle(img_d, (left, top), (right, bottom)
                          , (255, 0, 0), 4)
        else:
            cv2.rectangle(img_d, (left, top), (right, bottom)
                          , (0,255, 0), 4)
        i = 0
        for s,l in zip(scores,labels):
            cv2.rectangle(img_d, (left, top - w), (left + 450, top + w +2*w*i), (255, 255, 255), cv2.FILLED)
            i+=1
        i = 0
        for s,l in zip(scores,labels):
            text = '{0} - {1:.3f} {2}'.format(l, s,id2name[l])
            cv2.putText(img_d, text, (left, top+2*w*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 0), thickness=2)
            i+=1

    plt.imshow(img_d)
    #plt.figure(figsize=(28, 28))
    fig.savefig(save_file_path)

def test_on_query_image(fs_serv,test_img_fname,score_thresh=0.1,det_engines=1,figure_factor=2.2,FontScale=2.3):
    img = cv2.imread(test_img_fname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    dets_n = fs_serv.detect_on_image(img, score_thresh=score_thresh, det_engines=det_engines)
    #dets_n = get_dets_class_names(dets_n)

    test_img_basename = os.path.basename(test_img_fname)
    img_file_path = os.path.join(disp_folder, test_img_basename.replace('.jpg', '_disp_n.jpg'))
    disp_dets2(img, dets_n, img_file_path,figure_factor=figure_factor,FontScale=FontScale)

def display_RedHat_examples():

    image_set = ['logo1.jpg','logo2.jpg','logo3.jpg']
    fig = plt.figure(2)
    ff = 2
    fig.set_size_inches((ff*8.5,3*ff*11),forward=False)
    enrollment_root = '../data/Logo/logo_usecase_data/RedHat/RedHat_enrollment'
    for cnt,img_basename in enumerate(image_set):
        imgname = os.path.join(enrollment_root,img_basename)
        img = mpimg.imread(imgname)
        plt.subplot(1,3,cnt+1)
        plt.imshow(img)
        plt.axis('off')

def display_lacoste_examples():

    image_set = ['5cadb37d4b967f67d3047964.jpg','lacoste-logo-sweatshirt-white-p12654-72879_image.jpg','12686073-1-white.jpg']
    fig = plt.figure(2)
    ff = 2
    fig.set_size_inches((ff*8.5,3*ff*11),forward=False)
    enrollment_root = '../data/Logo/logo_usecase_data/lacoste/lacoste_enrollment/images'
    for cnt,img_basename in enumerate(image_set):
        imgname = os.path.join(enrollment_root,img_basename)
        img = mpimg.imread(imgname)
        plt.subplot(1,3,cnt+1)
        plt.imshow(img)
        plt.axis('off')