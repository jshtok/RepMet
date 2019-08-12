# -----------------------------------------------------------
# Part of RepMet codebase
# Joseph Shtok josephs@il.ibm.com, CVAR team, IBM Research AI
# -----------------------------------------------------------

"""
LOGO database
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
from utils.miscellaneous import assert_folder

class Logo(IMDB):
    def __init__(self, image_set, root_path, dataset_path, result_path=None, mask_size=-1,\
                 binary_thresh=None, categ_index_offs=0, per_category_epoch_max=0, \
                 classes_list_fname='Logo_classes_206.txt',\
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

        self.classes_list_fname = os.path.join(dataset_path,classes_list_fname)
        if image_set=='mw206_train':
            database_csv_fname = 'Logo_train_db_mw206a.csv'
            imageset_path = os.path.join(dataset_path, 'GT_imdata')
        if image_set=='mw206_test':
            database_csv_fname = 'Logo_test_db_mw206a.csv'
            imageset_path = os.path.join(dataset_path, 'GT_imdata')
        if image_set == 'mw206_short':
            database_csv_fname = 'Logo_short_db_mw206a.csv'
            imageset_path = os.path.join(dataset_path, 'GT_imdata')
        self.database_csv_fname = os.path.join(dataset_path,database_csv_fname)

        # year = image_set.split('_')[0]
        # image_set = image_set[len(year) + 1 : len(image_set)]
        # base_modifier = self.ds_main_name.lower().replace('-','')
        # modifier = base_modifier
        # super(JES, self).__init__('imagenet_' + modifier, image_set, root_path, dataset_path, result_path)  # set self.name
        # self.year = year
        self.image_set = image_set
        self.imageset_path = imageset_path
        self.name = 'Logo_'+image_set
        with open(self.classes_list_fname, 'r') as fid:
            self.classes = [x.strip() for x in fid.readlines()]
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(1,self.num_classes+1)))

        self.image_set_index =0
        self._result_path = '' #TODO: find out why do I need this field here

        roidb = self.gt_roidb()
        self.num_images = len(roidb)

        # self.image_set_index = self.load_image_set_index()
        # self.num_images = len(self.image_set_index)
        # print 'num_images', self.num_images
        # self.mask_size = mask_size
        # self.binary_thresh = binary_thresh
        #
        # self.config = {'comp_id': 'comp4',
        #                'use_diff': False,
        #                'min_size': 2}

    # def load_image_set_index(self):
    #     """
    #     find out which indexes correspond to given image set (train or val)
    #     :return:
    #     """
    #     image_set_index_file = os.path.join(self.dataset_path, 'ImageSets', self.image_set + '.txt')
    #     assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
    #     with open(image_set_index_file) as f:
    #         image_set_index = [x.split(' ')[0].strip() for x in f.readlines()]
    #
    #     filter_string = 'ILSVRC2013_train_extra'
    #     image_set_index = [x for x in image_set_index if not x.startswith(filter_string)]
    #
    #     return image_set_index

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
            self.roidb = roidb
            return roidb

# <produce roidb ---------------------------------------------------------------
        with open(self.database_csv_fname,'r') as fid:
            database_csv = [x for x in fid.readlines()]
        roidb = []
        for line in database_csv:
            line = line.strip()
            fields = line.split(',')

            image_name = fields[11]
            image_path, image_name = os.path.split(image_name)
            img_path = os.path.join(self.imageset_path,image_name)

            class_num = self._class_to_ind[fields[1]]
            class_idx = class_num-1
            TLWH = np.array(fields[4:8]).astype(np.int32)
            top = TLWH[0]
            left = TLWH[1]
            right = left+TLWH[2]
            bottom = top+TLWH[3]
            BBs = np.expand_dims([left, top, right, bottom],axis=0)
            oneHot = np.zeros((1, self.num_classes), dtype=np.float32)
            oneHot[0, class_idx] = 1

            append = False
            for nRoi, entry in enumerate(roidb):
                if entry['image']==img_path:
                    append = True
                    append_idx = nRoi
            if append:
                roidb[append_idx]['boxes'] =np.concatenate((roidb[append_idx]['boxes'],BBs),axis=0)
                roidb[append_idx]['gt_classes']+=[class_num]
                roidb[append_idx]['gt_overlaps'] = np.concatenate((roidb[append_idx]['gt_overlaps'],oneHot), axis=0)
                continue
            #im = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            roidb.append({
                'boxes': BBs,
                'flipped': False,
                'gt_classes': [class_num],
                'gt_overlaps': oneHot,
                'width': int(fields[12]),
                'height': int(fields[13]),
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

    # def gt_segdb(self):
    #     """
    #     return ground truth image regions database
    #     :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    #     """
    #     # cache_file = os.path.join(self.cache_path, self.name + '_gt_segdb.pkl')
    #     # if os.path.exists(cache_file):
    #     #     with open(cache_file, 'rb') as fid:
    #     #         segdb = cPickle.load(fid)
    #     #     print '{} gt segdb loaded from {}'.format(self.name, cache_file)
    #     #     return segdb
    #     #
    #     # gt_segdb = [self.load_imagenet_segmentation_annotation(index) for index in self.image_set_index]
    #     # with open(cache_file, 'wb') as fid:
    #     #     cPickle.dump(gt_segdb, fid, cPickle.HIGHEST_PROTOCOL)
    #     # print 'wrote gt segdb to {}'.format(cache_file)
    #     #
    #     # return gt_segdb
    #     raise NotImplementedError
    #
    # def load_imagenet_annotation(self, index):
    #     """
    #     for a given index, load image and bounding boxes info from XML file
    #     :param index: index of a specific image
    #     :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    #     """
    #     import xml.etree.ElementTree as ET
    #     roi_rec = dict()
    #     roi_rec['image'] = self.image_path_from_index(index)
    #
    #     filename = os.path.join(self.dataset_path, 'Annotations', self.image_set, index + '.xml')
    #
    #     if not os.path.isfile(filename):
    #         raise NotImplementedError
    #
    #         from PIL import Image
    #         im = Image.open(self.image_path_from_index(index))
    #         width, height = im.size
    #         roi_rec['height'] = height
    #         roi_rec['width'] = width
    #         gt_classe_name = index.split('/')[0]
    #         cls = self._wnid_to_ind[gt_classe_name.lower().strip()]
    #         boxes = np.array([[0,0,width,height]], dtype=np.uint16)
    #         gt_classes = np.array([cls], dtype=np.int32)
    #         overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
    #         overlaps[cls] = 1
    #     else:
    #         tree = ET.parse(filename)
    #         size = tree.find('size')
    #         roi_rec['height'] = float(size.find('height').text)
    #         roi_rec['width'] = float(size.find('width').text)
    #         #im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
    #         #assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']
    #
    #         objs = tree.findall('object')
    #         if not self.config['use_diff']:
    #             non_diff_objs = [obj for obj in objs if (obj.find('difficult') is None) or int(obj.find('difficult').text) == 0]
    #             objs = non_diff_objs
    #         num_objs = len(objs)
    #
    #         boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    #         gt_classes = np.zeros((num_objs), dtype=np.int32)
    #         overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    #
    #         # class_to_index = dict(zip(self.classes, range(self.num_classes)))
    #         # Load object bounding boxes into a data frame.
    #         for ix, obj in enumerate(objs):
    #             bbox = obj.find('bndbox')
    #             # Make pixel indexes 0-based
    #             x1 = np.maximum(float(bbox.find('xmin').text) - 1,0)
    #             y1 = np.maximum(float(bbox.find('ymin').text) - 1,0)
    #             x2 = np.maximum(float(bbox.find('xmax').text) - 1,0)
    #             y2 = np.maximum(float(bbox.find('ymax').text) - 1,0)
    #
    #             if (x2<=x1) or (y2<=y1):
    #                 assert (x2<=x1) or (y2<=y1), 'tlc>brc'
    #
    #             from PIL import Image
    #             im = Image.open(self.image_path_from_index(index))
    #             width, height = im.size
    #             if (roi_rec['width'] != width) or (roi_rec['height'] != height):
    #                 assert (roi_rec['width'] != width) or (roi_rec['height'] != height), 'wrongly recorded image size'
    #
    #             if (y2>=height) or (x2>=width):
    #                 assert (y2>=height) or (x2>=width), 'bb exceeds image boundaries'
    #
    #             cls = self._wnid_to_ind[str(obj.find("name").text).lower().strip()]
    #             # cls = class_to_index[obj.find('name').text.lower().strip()]
    #
    #             boxes[ix, :] = [x1, y1, x2, y2]
    #             gt_classes[ix] = cls
    #             overlaps[ix, cls] = 1.0
    #
    #     roi_rec.update({'boxes': boxes,
    #                     'gt_classes': gt_classes,
    #                     'gt_overlaps': overlaps,
    #                     'max_classes': overlaps.argmax(axis=1),
    #                     'max_overlaps': overlaps.max(axis=1),
    #                     'flipped': False,
    #                     'per_category_epoch_max': self.per_category_epoch_max})
    #     return roi_rec
    #
    # def load_selective_search_roidb(self, gt_roidb):
    #     """
    #     turn selective search proposals into selective search roidb
    #     :param gt_roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    #     :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
    #     """
    #     # import scipy.io
    #     # matfile = os.path.join(self.root_path, 'selective_search_data', self.name + '.mat')
    #     # assert os.path.exists(matfile), 'selective search data does not exist: {}'.format(matfile)
    #     # raw_data = scipy.io.loadmat(matfile)['boxes'].ravel()  # original was dict ['images', 'boxes']
    #     #
    #     # box_list = []
    #     # for i in range(raw_data.shape[0]):
    #     #     boxes = raw_data[i][:, (1, 0, 3, 2)] - 1  # pascal voc dataset starts from 1.
    #     #     keep = unique_boxes(boxes)
    #     #     boxes = boxes[keep, :]
    #     #     keep = filter_small_boxes(boxes, self.config['min_size'])
    #     #     boxes = boxes[keep, :]
    #     #     box_list.append(boxes)
    #     #
    #     # return self.create_roidb_from_box_list(box_list, gt_roidb)
    #     raise NotImplementedError
    #
    # def selective_search_roidb(self, gt_roidb, append_gt=False):
    #     """
    #     get selective search roidb and ground truth roidb
    #     :param gt_roidb: ground truth roidb
    #     :param append_gt: append ground truth
    #     :return: roidb of selective search
    #     """
    #     # cache_file = os.path.join(self.cache_path, self.name + '_ss_roidb.pkl')
    #     # if os.path.exists(cache_file):
    #     #     with open(cache_file, 'rb') as fid:
    #     #         roidb = cPickle.load(fid)
    #     #     print '{} ss roidb loaded from {}'.format(self.name, cache_file)
    #     #     return roidb
    #     #
    #     # if append_gt:
    #     #     print 'appending ground truth annotations'
    #     #     ss_roidb = self.load_selective_search_roidb(gt_roidb)
    #     #     roidb = IMDB.merge_roidbs(gt_roidb, ss_roidb)
    #     # else:
    #     #     roidb = self.load_selective_search_roidb(gt_roidb)
    #     # with open(cache_file, 'wb') as fid:
    #     #     cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    #     # print 'wrote ss roidb to {}'.format(cache_file)
    #     #
    #     # return roidb
    #     raise NotImplementedError
    #
    # def load_imagenet_segmentation_annotation(self, index):
    #     """
    #     for a given index, load image and bounding boxes info from XML file
    #     :param index: index of a specific image
    #     :return: record['seg_cls_path', 'flipped']
    #     """
    #     # import xml.etree.ElementTree as ET
    #     # seg_rec = dict()
    #     # seg_rec['image'] = self.image_path_from_index(index)
    #     # size = cv2.imread(seg_rec['image']).shape
    #     # seg_rec['height'] = size[0]
    #     # seg_rec['width'] = size[1]
    #     #
    #     # seg_rec['seg_cls_path'] = self.segmentation_path_from_index(index)
    #     # seg_rec['flipped'] = False
    #     #
    #     # return seg_rec
    #     raise NotImplementedError
    #

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


    def evaluate_detections(self,detections,logger,display=False,display_folder = '/dccstor/jsdata1/dev/RepMet/output/Logo_mw206/disp_test'):
    # detections: list of length #classes (+bkgnd). Each entry is a list of size #<test images>, of arrays of 5-row detections, some are empty.
        from utils.miscellaneous import strip_special_chars,configure_logging
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
                disp_classes = [strip_special_chars(s) for s in self.classes]
                for i, d in enumerate(disp_classes):
                    if d.find('Sainsbury') == 0:
                        disp_classes[i]='Sainsbury'
                gt_classes = [strip_special_chars(s) for s in entry['gt_names']]
                show_dets_gt_boxes(im, dets_reflow[img_num][1:],disp_classes, entry['boxes'],gt_classes, scale=1.0, FS=8,LW=1.5, save_file_path=os.path.join(display_folder,'disp_{0}.png'.format(im_fname)))

        stats.print_perf(logger, prefix='[1]')
        my_logger = configure_logging('/dccstor/jsdata1/dev/RepMet/data/JES_pilot/tmp_logger.log')
        stats.print_perf(my_logger, prefix='[2]')





    # def evaluate_detections(self, detections):
    #     """
    #     top level evaluations
    #     :param detections: result matrix, [bbox, confidence]
    #     :return: None
    #     """
    #     # make all these folders for results
    #     result_dir = os.path.join(self.result_path, 'results')
    #     if not os.path.exists(result_dir):
    #         os.mkdir(result_dir)
    #     # year_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year)
    #     # if not os.path.exists(year_folder):
    #     #     os.mkdir(year_folder)
    #     res_file_folder = os.path.join(self.result_path, 'results',  'CLS-LOC')
    #     if not os.path.exists(res_file_folder):
    #         os.mkdir(res_file_folder)
    #     self.image_set_index = range(len(detections[0]))
    #     self.write_Logo_results(detections)
    #     info = self.do_python_eval()
    #     return info
    #     #aise NotImplementedError
















    # def evaluate_segmentations(self, pred_segmentations=None):
    #     """
    #     top level evaluations
    #     :param pred_segmentations: the pred segmentation result
    #     :return: the evaluation results
    #     """
    #     # make all these folders for results
    #     # if not (pred_segmentations is None):
    #     #     self.write_pascal_segmentation_result(pred_segmentations)
    #     #
    #     # info = self._py_evaluate_segmentation()
    #     # return info
    #     raise NotImplementedError
    #
    # def write_imagenet_segmentation_result(self, pred_segmentations):
    #     """
    #     Write pred segmentation to res_file_folder
    #     :param pred_segmentations: the pred segmentation results
    #     :param res_file_folder: the saving folder
    #     :return: [None]
    #     """
    #     # result_dir = os.path.join(self.result_path, 'results')
    #     # if not os.path.exists(result_dir):
    #     #     os.mkdir(result_dir)
    #     # year_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year)
    #     # if not os.path.exists(year_folder):
    #     #     os.mkdir(year_folder)
    #     # res_file_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year, 'Segmentation')
    #     # if not os.path.exists(res_file_folder):
    #     #     os.mkdir(res_file_folder)
    #     #
    #     # result_dir = os.path.join(self.result_path, 'results', 'VOC' + self.year, 'Segmentation')
    #     # if not os.path.exists(result_dir):
    #     #     os.mkdir(result_dir)
    #     #
    #     # pallete = self.get_pallete(256)
    #     #
    #     # for i, index in enumerate(self.image_set_index):
    #     #     segmentation_result = np.uint8(np.squeeze(np.copy(pred_segmentations[i])))
    #     #     segmentation_result = PIL.Image.fromarray(segmentation_result)
    #     #     segmentation_result.putpalette(pallete)
    #     #     segmentation_result.save(os.path.join(result_dir, '%s.png'%(index)))
    #     raise NotImplementedError
    #
    # def get_pallete(self, num_cls):
    #     """
    #     this function is to get the colormap for visualizing the segmentation mask
    #     :param num_cls: the number of visulized class
    #     :return: the pallete
    #     """
    #     n = num_cls
    #     pallete = [0]*(n*3)
    #     for j in xrange(0,n):
    #             lab = j
    #             pallete[j*3+0] = 0
    #             pallete[j*3+1] = 0
    #             pallete[j*3+2] = 0
    #             i = 0
    #             while (lab > 0):
    #                     pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
    #                     pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
    #                     pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
    #                     i = i + 1
    #                     lab >>= 3
    #     return pallete
    #
    # def get_confusion_matrix(self, gt_label, pred_label, class_num):
    #     """
    #     Calcute the confusion matrix by given label and pred
    #     :param gt_label: the ground truth label
    #     :param pred_label: the pred label
    #     :param class_num: the nunber of class
    #     :return: the confusion matrix
    #     """
    #     index = (gt_label * class_num + pred_label).astype('int32')
    #     label_count = np.bincount(index)
    #     confusion_matrix = np.zeros((class_num, class_num))
    #
    #     for i_label in range(class_num):
    #         for i_pred_label in range(class_num):
    #             cur_index = i_label * class_num + i_pred_label
    #             if cur_index < len(label_count):
    #                 confusion_matrix[i_label, i_pred_label] = label_count[cur_index]
    #
    #     return confusion_matrix
    #
    # def _py_evaluate_segmentation(self):
    #     """
    #     This function is a wrapper to calculte the metrics for given pred_segmentation results
    #     :param pred_segmentations: the pred segmentation result
    #     :return: the evaluation metrics
    #     """
    #     confusion_matrix = np.zeros((self.num_classes,self.num_classes))
    #     result_dir = os.path.join(self.result_path, 'results', 'Segmentation')
    #
    #     for i, index in enumerate(self.image_set_index):
    #         seg_gt_info = self.load_pascal_segmentation_annotation(index)
    #         seg_gt_path = seg_gt_info['seg_cls_path']
    #         seg_gt = np.array(PIL.Image.open(seg_gt_path)).astype('float32')
    #         seg_pred_path = os.path.join(result_dir, '%s.png'%(index))
    #         seg_pred = np.array(PIL.Image.open(seg_pred_path)).astype('float32')
    #
    #         seg_gt = cv2.resize(seg_gt, (seg_pred.shape[1], seg_pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    #         ignore_index = seg_gt != 255
    #         seg_gt = seg_gt[ignore_index]
    #         seg_pred = seg_pred[ignore_index]
    #
    #         confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, self.num_classes)
    #
    #     pos = confusion_matrix.sum(1)
    #     res = confusion_matrix.sum(0)
    #     tp = np.diag(confusion_matrix)
    #
    #     IU_array = (tp / np.maximum(1.0, pos + res - tp))
    #     mean_IU = IU_array.mean()
    #
    #     return {'meanIU':mean_IU, 'IU_array':IU_array}
    #
    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = assert_folder(os.path.join(self.result_path, 'results', 'Logo_'+self.image_set))

        filename = 'det_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path
    #     # raise NotImplementedError
    #
    def write_Logo_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind in range(len(all_boxes)): # correction for the case all_boxes contains only N first classes of self.classes. Joseph Shtok
            cls =self.classes[cls_ind]
        #for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} Logo results file'.format(cls)
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))
        #raise NotImplementedError
    #
    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        # JS: implement.

        info_str = ''
        # annopath = os.path.join(self.dataset_path, 'Annotations','CLS-LOC',self.image_set, '{0!s}.xml')
        annopath = os.path.join(self.dataset_path, 'Annotations', 'CLS-LOC', 'val', '{0!s}.xml') #TODO: remove - temporary fix to run val_partial
        imageset_file = os.path.join(self.dataset_path, 'ImageSets', 'CLS-LOC', self.image_set + '.txt')
        annocache = os.path.join(self.cache_path, self.name + '_annotations.pkl')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False #True if self.year == 'SDS' or int(self.year) < 2010 else False
        #print 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        #info_str += 'VOC07 metric? ' + ('Y' if use_07_metric else 'No')
        #info_str += '\n'
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            if not os.path.exists(filename): # JS Joseph Shtok: added for partial categories set
                continue
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.5 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(aps))
        # @0.7
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.7, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.7 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.7 = {:.4f}'.format(np.mean(aps))
        # @0.3
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, annopath, imageset_file, cls, annocache,
                                     ovthresh=0.3, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        print('Mean AP@0.3 = {:.4f}'.format(np.mean(aps)))
        info_str += 'Mean AP@0.3 = {:.4f}'.format(np.mean(aps))
        return info_str
       #raise NotImplementedError
