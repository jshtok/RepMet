# -----------------------------------------------------------
# Part of RepMet codebase
# Joseph Shtok josephs@il.ibm.com, CVAR team, IBM Research AI
# -----------------------------------------------------------

"""
Defects_poc database
This class loads ground truth notations from Defects PoC XML data format
and transform them into IMDB format. Selective search is used for proposals, see roidb function.
"""

import cPickle
import cv2
import os
import numpy as np
#import PIL
import scipy.io as sio

from imdb import IMDB
from imagenet_voc_eval import voc_eval, voc_eval_sds
from ds_utils import unique_boxes, filter_small_boxes
from utils.miscellaneous import assert_folder,configure_logging

class Defects(IMDB):
    def __init__(self, image_set, root_path, dataset_path, result_path=None, mask_size=-1,\
                 binary_thresh=None, categ_index_offs=0, per_category_epoch_max=0, \
                 classes_list_fname='', \
                 num_ex_per_class=''):
        """
        fill basic information to initialize imdb
        :param image_set: poc_train, poc_val, poc_test, pilot_*    # 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param dataset_path: data and results
        :return: imdb object
        """

        sis = image_set.split(':') # take just the first dataset in case a sequence is passed
        if len(sis) > 1:
            image_set = sis[0]
        self.per_category_epoch_max = per_category_epoch_max
        self.root_path = root_path
        self.dataset_path = dataset_path
        self.logger = configure_logging('Defects_logger.log')
        database_csv_fname = []
        if image_set[0:5]=='train':
            database_csv_fname = 'defects1_db.csv'

        self.database_csv_fname = os.path.join(dataset_path,database_csv_fname)
        self.classes_list_fname = os.path.join(dataset_path,classes_list_fname)

        self.image_set = image_set
        self.name = 'Defects_'+image_set
        with open(self.classes_list_fname, 'r') as fid:
            self.classes = [x.strip() for x in fid.readlines()]
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(1,self.num_classes+1)))
        self.image_set_index =0
    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            self.num_images = len(roidb)
            self.roidb = roidb
            return roidb


# <produce roidb ---------------------------------------------------------------
        with open(self.database_csv_fname,'r') as fid:
            database_csv = [x for x in fid.readlines()]
        roidb = []
        database_csv = np.unique(database_csv).tolist()
        for line in database_csv:
            line = line.strip()
            fields = line.split(',')
            if fields[1] not in self._class_to_ind.keys():
                continue

            # im_idx=1 if self.dataset_type == 'poc' else 8
            # left_idx = 5 if self.dataset_type == 'poc' else 4
            # top_idx = 4 if self.dataset_type == 'poc' else 5
            # imw_idx = 12 if self.dataset_type == 'poc' else 9
            # imh_idx = 13 if self.dataset_type == 'poc' else 10
            im_idx= 8
            left_idx =  4
            top_idx =  5
            imw_idx =  9
            imh_idx = 10

            image_name = fields[im_idx]
            if len(self.image_list) > 0 and image_name not in self.image_list:
                continue
            class_num = self._class_to_ind[fields[1]]
            class_name = fields[1]
            class_idx = class_num - 1
            left = int(fields[left_idx])
            top = int(fields[top_idx])
            width = int(fields[6])
            height = int(fields[7])

            if left <= 0:
                left = 1
            right = left + width
            if top <= 0:
                top = 1
            bottom = top + height

            img_width = int(fields[imw_idx])
            img_height = int(fields[imh_idx])
            if right>= img_width:
                right = img_width - 1
            if bottom>= img_height:
                bottom = img_height - 1

            # if self.dataset_type=='poc':
            #     image_name = fields[11]
            #     if len(self.image_list)>0 and image_name not in self.image_list:
            #         continue
            #     class_num = self._class_to_ind[fields[1]]
            #     class_name = fields[1]
            #     class_idx = class_num-1
            #     TLWH = np.array(fields[4:8]).astype(np.uint16)
            #     left = TLWH[1]
            #     if left<=0:
            #         left=1
            #     right = left+TLWH[2]
            #     top = TLWH[0]
            #     if top<=0:
            #         top=1
            #     bottom = top+TLWH[3]
            #
            #     img_width = int(fields[12])
            #     img_height = int(fields[13])
            #     if img_width>=right:
            #         right = img_width-1
            #     if img_height>=bottom:
            #         bottom = img_height-1
            #
            # if self.dataset_type == 'pilot':
            #     image_name = fields[8]
            #     if len(self.image_list)>0 and image_name not in self.image_list:
            #         continue
            #     class_num = self._class_to_ind[fields[1]]
            #     class_name = fields[1]
            #     class_idx = class_num - 1
            #     left = int(fields[4])
            #     top = int(fields[5])
            #     width = int(fields[6])
            #     height = int(fields[7])
            #
            #     if left<=0:
            #         left=1
            #     right = left + width
            #     if top<=0:
            #         top=1
            #     bottom = top + height
            #
            #     img_width = int(fields[9])
            #     img_height = int(fields[10])
            #     if img_width>=right:
            #         right = img_width-1
            #     if img_height>=bottom:
            #         bottom = img_height-1

            BBs = np.expand_dims([left, top, right, bottom],axis=0)
            oneHot = np.zeros((1, self.num_classes), dtype=np.float32)
            oneHot[0, class_idx] = 1
            img_path =  os.path.join(self.dataset_path, 'images', image_name)
            append = False
            for nImg, entry in enumerate(roidb):
                if entry['image']==img_path:
                    append = True
                    append_idx = nImg
            if append:
                roidb[append_idx]['boxes'] =np.concatenate((roidb[append_idx]['boxes'],BBs),axis=0)
                roidb[append_idx]['gt_classes']+=[class_num]
                roidb[append_idx]['gt_names']+=[class_name]
                roidb[append_idx]['gt_overlaps'] = np.concatenate((roidb[append_idx]['gt_overlaps'],oneHot), axis=0)
                continue
            #im = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            roidb.append({
                'boxes': BBs,
                'flipped': False,
                'gt_classes': [class_num],
                'gt_names':[class_name],
                'gt_overlaps': oneHot,
                'width': int(img_width),
                'height': int(img_height),
                'image': img_path,
                #'max_classes': class_num
                #'max_overlaps': np.ones((BBs.shape[0], 1), dtype=np.float32),
                #'aug_gen': aug_gen
            })

        for entry in roidb:
            entry['gt_classes'] = np.asarray(entry['gt_classes'])
            entry['max_classes'] = entry['gt_classes']
            entry['max_overlaps'] = np.ones((entry['boxes'].shape[0], 1), dtype=np.float32),

        self.num_images = len(roidb)
        self.roidb = roidb

        # >produce roidb ---------------------------------------------------------------

        # gt_roidb = []
        # for ii, index in enumerate(self.image_set_index):
        #     if (ii % 1000) == 0:
        #         print('Processing image {0} of {1}'.format(ii,len(self.image_set_index)))
        #     gt_roidb.append(self.load_imagenet_annotation(index))
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self.roidb, fid, cPickle.HIGHEST_PROTOCOL) # gt_roidb
        print 'wrote gt roidb to {}'.format(cache_file)
        return self.roidb

    def evaluate_detections(self,detections,logger):
    # detections: list of length #classes (+bkgnd). Each entry is a list of size #<test images>, of arrays of 5-row detections, some are empty.
        from utils.PerfStats import PerfStats
        ovthresh = 0.5
        Nclasses = len(detections)
        Nimages = len(self.roidb)
        epi_cats = range(1, Nclasses+1 )
        stats = PerfStats(Nclasses=Nclasses)
        dets_reflow = [ [] for _ in range(Nimages)]
        for img_num in range(Nimages):
            for cls in range(Nclasses):
                if len(detections[cls][img_num]) > 0:
                    dets_reflow[img_num].append( detections[cls][img_num])
                else:
                    dets_reflow[img_num].append(np.zeros((0,5)))

        for img_num, entry in enumerate(self.roidb):
            #stats.comp_epi_stats_m(dets_reflow[img_num][1:],entry['boxes'],entry['gt_classes'], epi_cats,ovthresh)
            stats.comp_epi_stats_m(dets_reflow[img_num], entry['boxes'], entry['gt_classes'], epi_cats, ovthresh)

        my_logger = configure_logging('/dccstor/jsdata1/dev/RepMet/data/Defects_pilot/tmp_logger.log')
        stats.print_perf(my_logger, prefix='')
