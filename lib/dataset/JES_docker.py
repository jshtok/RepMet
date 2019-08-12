# -----------------------------------------------------------
# Part of RepMet codebase
# Joseph Shtok josephs@il.ibm.com, CVAR team, IBM Research AI
# -----------------------------------------------------------

"""
JES_poc database
This class loads ground truth notations from JES PoC XML data format
and transform them into IMDB format. Selective search is used for proposals, see roidb function.
"""

import cPickle
import cv2
import os
import numpy as np
import PIL
import scipy.io as sio

from imdb import IMDB
from imagenet_voc_eval import voc_eval, voc_eval_sds
from ds_utils import unique_boxes, filter_small_boxes
from utils.miscellaneous import assert_folder,configure_logging

class JES(IMDB):
    def __init__(self, image_set, root_path, dataset_path, result_path=None, mask_size=-1,\
                 binary_thresh=None, categ_index_offs=0, per_category_epoch_max=0, \
                 database_csv='', classes_list_fname='',\
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
        self.logger = configure_logging('JES_logger.log')
        database_csv_fname = database_csv
        image_list_fname = []
        if image_set[0:3]=='poc':
            database_csv_fname = 'train38_db_oc_fname.csv'
            #database_csv_fname = 'annotations_Aug7_NoRot_38Prod.csv'
            self.dataset_type = 'poc'

        if image_set == 'few_shot_poc':
            database_csv_fname = 'JES_PoC_fs_db.csv'
            self.dataset_type = 'poc'

        if image_set[0:11]=='pilot_foods':
            database_csv_fname = 'all_GT.csv_converted_Feb24.csv'
            self.dataset_type = 'pilot'

        if image_set[0:11]=='pilot_f_nto':
            database_csv_fname = 'all_GT.csv_converted_Feb24_nto.csv'
            self.dataset_type = 'pilot'

        if image_set[0:12]=='pilot_plates':
            database_csv_fname = 'all_GT_CNT_Feb24_merged.csv'
            self.dataset_type = 'pilot'

        if image_set[0:17]=='pilot_MultiPlates':
            database_csv_fname = 'all_GT_CNT_Feb24.csv'
            self.dataset_type = 'pilot'

        if image_set=='pilot_Apr17_enroll':
            database_csv_fname ='pilot_Apr17_enroll.csv'


        if image_set=='pilot_Apr17_17_enroll':
            database_csv_fname ='pilot_Apr17_17_enroll.csv'

        if image_set=='pilot_Apr17_17_test':
            database_csv_fname ='pilot_Apr17_17_test.csv'

        if image_set=='pilot_Apr17_test':
            database_csv_fname ='pilot_Apr17_test.csv'


        self.database_csv_fname = os.path.join(dataset_path,database_csv_fname)
        self.classes_list_fname = os.path.join(dataset_path,classes_list_fname)
        if image_set[0:9]=='poc_train' or image_set[0:7]=='poc_dev':
            image_list_fname = ['Aug7_NoRot_data_train_images.csv']
        if image_set=='poc_test':
            image_list_fname = ['Aug7_NoRot_data_test_images.csv']
        if image_set=='poc_val':
            image_list_fname = ['Aug7_NoRot_data_val_images.csv']

        if image_set=='poc_test_val':
            image_list_fname = ['Aug7_NoRot_data_test_images.csv','Aug7_NoRot_data_val_images.csv']
        if image_set=='poc_test_val_8':
            image_list_fname = ['Aug7_NoRot_data_test_val_images_8.csv']

        if image_set=='pilot_foods_train' or image_set=='pilot_plates_train' or image_set=='pilot_MultiPlates_train':
            image_list_fname = ['JES_pilot_train_images.txt']
        if image_set=='pilot_foods_test' or image_set=='pilot_plates_test'or image_set=='pilot_MultiPlates_test':
            image_list_fname = ['JES_pilot_test_images.txt']
        if image_set=='pilot_foods_test_short' or image_set=='pilot_MultiPlates_test_short':
            image_list_fname = ['JES_pilot_test_images_short.txt']

        image_list = []
        for list_fname in image_list_fname:
            full_list_fname = os.path.join(dataset_path,list_fname)
            with open(full_list_fname, 'r') as fid:
                image_list.extend([x.strip() for x in fid.readlines()])
            print('length of image_list: {0}'.format(len(image_list)))
        self.image_list = image_list
        self.image_set = image_set
        self.name = 'JES_'+image_set
        with open(self.classes_list_fname, 'r') as fid:
            self.classes = [x.strip() for x in fid.readlines()]
        self.num_classes = len(self.classes)+1
        self._class_to_ord = dict(zip(self.classes, xrange(1,self.num_classes)))

        self.image_set_index =0
        self._result_path = ''

    def gt_roidb(self, force_reload = False):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join('/vol1/output/cache/', self.name + '_gt_roidb.pkl')
        if not force_reload and os.path.exists(cache_file):
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
            if fields[1] not in self._class_to_ord.keys():
                self.logger.info('warning: category {0} in image {1} is not on the list. Ignoring.'.format(fields[1],fields[8]))
                continue

            im_idx= 8
            left_idx =  4
            top_idx =  5
            imw_idx =  9
            imh_idx = 10

            image_name = fields[im_idx]
            if len(self.image_list) > 0 and image_name not in self.image_list:
                continue
            class_ord = self._class_to_ord[fields[1]]
            class_name = fields[1]
            class_idx = class_ord - 1
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
                roidb[append_idx]['gt_classes']+=[class_ord]
                roidb[append_idx]['gt_names']+=[class_name]
                roidb[append_idx]['gt_overlaps'] = np.concatenate((roidb[append_idx]['gt_overlaps'],oneHot), axis=0)
                continue
            #im = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            roidb.append({
                'boxes': BBs,
                'flipped': False,
                'gt_classes': [class_ord],
                'gt_names':[class_name],
                'gt_overlaps': oneHot,
                'width': int(img_width),
                'height': int(img_height),
                'image': img_path,
                #'max_classes': class_ord
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

    def export_dets_B2C(self,q_dets_novl,dets_export_fname,ord2name):
        with open(dets_export_fname, 'w') as fid_w:
            #fid_w.write('%s\n' % format('<class number>;<score>;<Left>;<Top>;<Right>;<Bottom>;<Cat name>'))
            for idx, entry in enumerate(q_dets_novl):
                cat = idx+1
                cat_name = ord2name[cat]
                for det in entry:
                    tline = '{0};{1:.3f};{2};{3};{4};{5}'.format(cat_name, det[4], int(det[0]),int(det[1]),\
                                       int(det[2]),int(det[3]))
                    fid_w.write('%s\n' % format(tline))

    def evaluate_detections(self,detections,display=False,display_folder = '/dccstor/jsdata1/dev/RepMet/output/JES_pilot/disp_foods_main_ep8_corr_nms'):
    # detections: list of length #classes (+bkgnd). Each entry is a list of size #<test images>, of arrays of 5-row detections, some are empty.
        from utils.show_boxes import show_boxes, show_dets_gt_boxes
        from utils.PerfStats import PerfStats
        assert_folder(display_folder)
        ovthresh = 0.5
        score_thresh = 0.1
        Nclasses = len(detections)
        Nimages = len(self.roidb)
        epi_cats = range(1, Nclasses)
        stats = PerfStats(Nclasses=Nclasses)
        dets_reflow = [ [] for _ in range(Nimages)]
        for img_num in range(Nimages):
            for cls_idx in range(Nclasses):
                if len(detections[cls_idx][img_num]) > 0:
                    valid_dets = detections[cls_idx][img_num][np.where(detections[cls_idx][img_num][:,4]>score_thresh)]
                    dets_reflow[img_num].append(valid_dets)
                else:
                    dets_reflow[img_num].append(np.zeros((0,5)))
        name2ord = {}
        ord2name = {}
        for idx, name in enumerate(self.classes):
            ord = idx+1
            name2ord[name] = ord
            ord2name[ord] = name

        for img_num, entry in enumerate(self.roidb):
            im_path, im_fname = os.path.split(entry['image'])
            dets_export_fname = os.path.join(display_folder,im_fname[:-4]+'.txt')
            #if im_fname[-8:-4]=='_top':
            self.export_dets_B2C(dets_reflow[img_num][1:],dets_export_fname,ord2name)

            stats.comp_epi_stats_m(dets_reflow[img_num][1:],entry['boxes'],entry['gt_classes'], epi_cats,ovthresh)
            if display:
                im = cv2.imread(entry['image'])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im_path, im_fname = os.path.split(entry['image'])
                show_dets_gt_boxes(im, dets_reflow[img_num][1:], self.classes, entry['boxes'],entry['gt_names'], scale=1.0, FS=8,LW=1.5, save_file_path=os.path.join(display_folder,'disp_{0}.png'.format(im_fname)))

        my_logger = configure_logging('/dccstor/jsdata1/dev/RepMet/data/JES_pilot/tmp_logger.log')
        stats.print_perf(my_logger, prefix='')

