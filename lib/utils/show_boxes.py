# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
plt.switch_backend('agg')
from random import random as rand

def show_detsB_gt_boxes(im, dets_B,  gt_boxes, gt_classes, scale = 1.0, save_file_path='temp_det_gt.png'):
    import matplotlib.pyplot as plt
    from random import random as rand
    fig = plt.figure(1)
    FS = 22
    fig.set_size_inches((2 * 8.5, 1 * 11), forward=False)

    plt.subplot(121)
    plt.cla()
    plt.axis("off")
    plt.imshow(im)


    for det in dets_B:
        det_row = det[0]
        cat_name = det[1]
        bbox = det_row[:4] * scale
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)
        score = det_row[-1]
        plt.gca().text(bbox[0], bbox[1],
                       '{:s} {:.3f}'.format(cat_name, score),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=FS, color='white')

    plt.subplot(122)
    plt.cla()
    plt.axis("off")
    plt.imshow(im)

    for cls_idx, cls_name in enumerate(gt_classes):
        bbox = gt_boxes[cls_idx]
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=3.5)
        plt.gca().add_patch(rect)

        plt.gca().text(bbox[0], bbox[1],
                       '{:s}'.format(cls_name),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=FS, color='white')

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(save_file_path, bbox_inches = 'tight',  pad_inches = 0)
    plt.close(fig)

def show_detsB_boxes(im, dets_B, scale = 1.0, save_file_path='temp.png'):
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for det in dets_B:
        det_row = det[0]
        cat_name = det[2]
        bbox = det_row[:4] * scale
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
    fig.savefig(save_file_path)

def show_dets_crops(im,dets,classes,scale=1.0,marg=0,save_file_path='temp_det_gt.png'):
    fig = plt.figure(2)
    plt.cla()
    fig.set_size_inches((8.5, 2 * 11), forward=False)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for idet,det in enumerate(cls_dets):
            plt.cla()
            bbox = det[:4] * scale
            #rect = im[int(bbox[0])-marg:int(bbox[2])+marg,int(bbox[1])-marg:int(bbox[3])+marg,:]
            rect = im[int(bbox[1]) - marg:int(bbox[3]) + marg, int(bbox[0]) - marg:int(bbox[2]) + marg, :]
            plt.imshow(rect)
            score = det[-1]
            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                hspace=0, wspace=0)
            plt.margins(0, 0)
            fig.savefig(save_file_path[:-4]+'_{0:.3f}_{1}.jpg'.format(score,idet), bbox_inches='tight', pad_inches=0)

def show_dets_gt_boxes(im, dets,classes,  gt_boxes,gt_classes, scale = 1.0,FS=22,LW=3.5, save_file_path='temp_det_gt.png'):
    import matplotlib.pyplot as plt
    import numpy as np
    from random import random as rand
    from random import randint
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 1 * 11), forward=False)

    plt.subplot(121)
    plt.cla()
    plt.axis("off")
    plt.imshow(im)


    for cls_dets, cls_name in zip(dets,classes):
        scores = []
        if len(cls_dets)==0:
            continue
        for det in cls_dets:
            scores+=[det[-1]]
        ord  = np.argsort(scores)
        cls_dets = cls_dets[ord]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=LW)
            plt.gca().add_patch(rect)
            score = det[-1]
            corner = randint(1, 2)
            if corner == 1:
                x0 = bbox[0]; y0 = bbox[1]
            if corner == 2:
                x0 = bbox[0]; y0 = bbox[3]
            # if corner == 3:
            #   x0 = bbox[2]; y0 = bbox[1]
            # if corner == 4:
            #     x0 = bbox[2]; y0 = bbox[3]

            plt.gca().text(x0,y0,
                           '{:s} {:.3f}'.format(cls_name, score),
                           bbox=dict(facecolor=color, alpha=0.6), fontsize=FS, color='white')

    plt.subplot(122)
    plt.cla()
    plt.axis("off")
    plt.imshow(im)

    for cls_idx, cls_name in enumerate(gt_classes):
        bbox = gt_boxes[cls_idx]
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=LW)
        plt.gca().add_patch(rect)

        plt.gca().text(bbox[0], bbox[1],
                       '{:s}'.format(cls_name),
                       bbox=dict(facecolor=color, alpha=0.6), fontsize=FS, color='white')

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig(save_file_path, bbox_inches = 'tight',  pad_inches = 0)
    plt.close(fig)

def show_boxes(im, dets, classes, scale = 1.0, save_file_path='temp.png'):
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for cls_dets, cls_name in zip(dets,classes):
    # for cls_idx, cls_name in enumerate(classes):
    #     cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=16, color='white')

               # print('{:s}: {:.3f}'.format(cls_name, score))

    #plt.show()

    fig.savefig(save_file_path)
    plt.close(fig)
    return im

def show_gt_boxes(im, boxes, classes, scale = 1.0, save_file_path='temp.png'):
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for cls_idx, cls_name in enumerate(classes):
        bbox = boxes[cls_idx]
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)

        plt.gca().text(bbox[0], bbox[1],
                       '{:s}'.format(cls_name),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=16, color='white')

               # print('{:s}: {:.3f}'.format(cls_name, score))

    #plt.show()

    fig.savefig(save_file_path)
    plt.close(fig)
    return im

def show_train_and_rois(im, box_tr, rois,save_file_path):
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)

    color = (0,1,0)
    for bbox in rois:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=1.0)
        plt.gca().add_patch(rect)

    color = (1,0,0)
    bbox = box_tr
    rect = plt.Rectangle((bbox[0], bbox[1]),
                         bbox[2] - bbox[0],
                         bbox[3] - bbox[1], fill=False,
                         edgecolor=color, linewidth=2.5)
    plt.gca().add_patch(rect)

    fig.savefig(save_file_path)

def show_just_boxes(im, boxes, scale = 1.0, save_file_path='temp.png'):
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for bbox in enumerate(boxes):
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)

    fig.savefig(save_file_path)

    return im
