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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def display_few_shot_examples():
    data_root = '/opt/DNN/dataset/retail/retail_test'
    image_set = ['0de3ec888566edc2.jpg','2e90529bcb43f44c.jpg',
                 '15a8f82801fe4911.jpg','34caf4cf9ae0ae92.jpg']
    nrows = 1 #int(np.sqrt(len(image_set)))
    ncols = np.ceil(len(image_set)/nrows)
    print('nrows={0},ncols={1}'.format(nrows, ncols))
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
    data_root = '/opt/DNN/dataset/retail/retail_test'
    imgname_left = os.path.join(data_root, '0de3ec888566edc2.jpg')
    imgname_top = os.path.join(data_root, '2e90529bcb43f44c.jpg')
    imgname_right = os.path.join(data_root, '15a8f82801fe4911.jpg')
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

import glob
import os
import json
from PIL import Image
import csv
import argparse


def oid2coco():
    img_dir = '/opt/DNN/dataset/openimage/openimage_train'
    boxes_csv = '/opt/DNN/dataset/openimage/retail.csv'
    json_file = '/opt/DNN/dataset/openimage/annotations/instances_openimage_train.json'
    classes_csv = '/opt/DNN/linkdata/eval_dataset/open_images/metadata/class-descriptions-boxable.csv'
    with open(classes_csv, 'r') as classes_file:
        classes = csv.reader(classes_file)
        class_dict = dict((rows[0], rows[1]) for rows in classes)
        classes_keys = class_dict.keys()
        classes_values = class_dict.values()

    images, anns, categories = [], [], []
    img_paths = [x for x in glob.glob(img_dir + '/*.jpg')]
    #num_imgs = len(img_paths)
    i = 1
    for img_path in sorted(img_paths):
        img = Image.open(img_path)
        width, height = img.size
        _, img_name = os.path.split(img_path)
        dic = {'file_name': img_name, 'id': i, 'height': height, 'width': width}
        images.append(dic)
        i += 1

    ann_index = 1
    i = 0
    with open(boxes_csv, "r") as boxes:
        lines = csv.reader(boxes)
        file_name_last_line = ""
        for line in lines:
            if line[0] == "ImageID":
                continue
            file_name = line[0] + '.jpg'
            full_image_path = os.path.join(img_dir, file_name)
            if os.path.exists(full_image_path) is not True:
                continue
            if file_name != file_name_last_line:
                i += 1
            img = Image.open(full_image_path)
            width, height = img.size
            xmin = float(line[4])
            xmax = float(line[5])
            ymin = float(line[6])
            ymax = float(line[7])
            area = int(((xmax - xmin) * width) * ((ymax - ymin) * height))
            poly = []
            bbox = [int(xmin * width), int(ymin * height), int((xmax - xmin) * width), int((ymax - ymin) * height)]
            category = line[2]
            cat_id = classes_keys.index(category)
            dic2 = {'segmentation': poly, 'area': area, 'iscrowd': 0, 'image_id': i, 'bbox': bbox,
                    'category_id': cat_id, 'id': ann_index, 'ignore': 0}
            anns.append(dic2)
            file_name_last_line = file_name
            ann_index += 1

    for cate in classes_values:
        cat = {'supercategory': 'none', 'id': classes_values.index(cate), 'name': cate}
        categories.append(cat)

    data = {'images': images, 'type': 'instances', 'annotations': anns, 'categories': categories}
    with open(json_file, 'w') as outfile:
        json.dump(data, outfile)

    with open(os.path.split(json_file)[0]+'/train_cats_split.txt','w') as fid:
        fid.writelines("\n".join(classes_values))

from random import random as rand
def show_detsB_boxes(im, dets_B, scale = 1.0, save_file_path='temp.png'):
    fig = plt.figure(1)
    ff = 1.0
    fig.set_size_inches((ff * 8.5, ff * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for det in dets_B:
        det_row = det[0]
        cat_name = det[2]
        bbox = det_row[:4] * scale
 #       print(bbox)
        #bbox = [int(box) for box in bbox]
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)
        score = det_row[-1]
        plt.gca().text(bbox[0], bbox[1],
                       '{:s} {:.3f}'.format(cat_name, score),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=16, color='white')
    #fig.savefig(save_file_path)

def display_few_shot_test(benchmark,query_image):
    import shutil
    from FSD_engine import FSD_RepMet as DetectionEngine
    from config.bench_config import bcfg, update_bench_config
    import cv2
    update_bench_config('../experiments/bench_configs/openimage_3_5_10_1.yaml')
    #query_image = '/opt/DNN/linkdata/eval_dataset/open_images/images/train/8cef18e3ba4c1165.jpg'
    q_dets_B, q_dets_multi_B = benchmark.det_eng.detect_on_image(query_image,num_candidates=5,cand_ver=1)
    img = cv2.cvtColor(cv2.imread(query_image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION),
                    cv2.COLOR_BGR2RGB)
    prob = [item for item in q_dets_B if item[0][4]>0.2]
#    print(prob)
    save_file_path = '/opt/DNN/dataset/retail/openimage_3_5_10_1/temp.jpg'
    show_detsB_boxes(img, prob,save_file_path = save_file_path)
