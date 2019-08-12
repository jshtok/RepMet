# -----------------------------------------------------------
# Part of RepMet codebase
# Joseph Shtok josephs@il.ibm.com, CVAR team, IBM Research AI
# -----------------------------------------------------------

"""
JES_poc database
This class loads ground truth notations from SCENT XML data format
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

class SCENT(IMDB):
    def __init__(self, image_set, root_path, dataset_path, result_path=None, mask_size=-1,
                 binary_thresh=None, categ_index_offs=0, per_category_epoch_max=0,
                 classes_list_fname='SCENT_base_classes.txt',
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
        database_csv_fname = []
        if image_set[0:4]=='base':
            database_csv_fname = 'SCENT_train_db.csv'
            self.dataset_type = 'base'

        self.database_csv_fname = os.path.join(dataset_path,database_csv_fname)
        self.classes_list_fname = os.path.join(dataset_path,classes_list_fname)

        self.image_set = image_set
        self.name = 'SCENT_'+image_set
        with open(self.classes_list_fname, 'r') as fid:
            self.classes = [x.strip() for x in fid.readlines()]
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(1,self.num_classes+1)))

        self.image_set_index =0
        self._result_path = '' #TODO: find out why do I need this field here

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.dataset_path, 'Data', self.image_set, index + '.JPEG')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def segmentation_path_from_index(self, index):
        """
        given image index, find out the full path of segmentation class
        :param index: index of a specific image
        :return: full path of segmentation class
        """
        # seg_class_file = os.path.join(self.dataset_path, 'SegmentationClass', index + '.png')
        # assert os.path.exists(seg_class_file), 'Path does not exist: {}'.format(seg_class_file)
        # return seg_class_file
        raise NotImplementedError

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
        cntr=0
        for line in database_csv:
            cntr+=1
            if cntr%1000==0:
                print('processed {0} entries'.format(cntr))
            line = line.strip()
            fields = line.split(',')
            img_path = fields[6].replace('E:/data/SCENT/imgs/TrainingImages','/dccstor/jsdata1/data/SCENT/TrainingImages')
            class_name = fields[1]
            class_name = class_name.replace('Park.Shrubs','Parks.Shrubs')
            class_name = class_name.replace('Park.Tall vegetation', 'Parks.Tall vegetation')
            class_name = class_name.replace('Storm drain.', 'Storm drains.')
            width = int(fields[7])
            height = int(fields[8])
            class_num = self._class_to_ind[class_name]
            BBs = np.expand_dims(np.array(fields[2:6]).astype(np.uint16),axis=0)
            # left = LTRB[0]
            # top = LTRB[1]
            # right = LTRB[2]
            # bottom = LTRB[3]
            #BBs = np.expand_dims([left, top, right, bottom],axis=0)
            oneHot = np.zeros((1, self.num_classes), dtype=np.float32)
            oneHot[0, class_num-1] = 1

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
            # im = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # if im is None:
            #     print('image {0} not found'.format(img_path))
            roidb.append({
                'boxes': BBs,
                'flipped': False,
                'gt_classes': [class_num],
                'gt_names':[class_name],
                'gt_overlaps': oneHot,
                'width': width,# im.shape[1],
                'height': height, #im.shape[0],
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

