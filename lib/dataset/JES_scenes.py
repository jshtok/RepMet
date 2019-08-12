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
import copy

from imdb import IMDB
from imagenet_voc_eval import voc_eval, voc_eval_sds
from ds_utils import unique_boxes, filter_small_boxes
from FSD_common_lib import assert_folder,configure_logging,get_view

class JES_scenes(IMDB):
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
       
        self.per_category_epoch_max = per_category_epoch_max
        self.root_path = root_path
        self.dataset_path = dataset_path
        self.logger = configure_logging('JESscenes_logger.log')
        database_csv_fname = []
        if image_set == 'pilot_foods_train':
            database_csv_file = 'all_GT.csv_converted_Feb24.csv'
        if image_set == 'pilot_foods_nto_train':
            database_csv_file = 'all_GT.csv_converted_Feb24_nto.csv'



        self.database_csv_fname = os.path.join(dataset_path, database_csv_file)
        self.classes_list_fname = os.path.join(dataset_path, classes_list_fname)
        self.image_set = image_set
        self.name = 'JES_'+image_set
        
        with open(self.classes_list_fname, 'r') as fid:
            self.class_names = [x.strip() for x in fid.readlines()]
        self.num_classes = len(self.class_names)+1
        self.className_to_ord = dict(zip(self.class_names, xrange(1,self.num_classes)))

        self.image_set_index =0
        self.views_list = ['left','top','right']


    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index][boxes_view,gt_overlaps_view, 'gt_classes', 'gt_names', 'flipped',images_view]
        boxes_view = {'top': boxes_top, 'right': boxes_right, 'left': boxes_left}
        gt_overlaps_view, images_view - same structure 
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_scenedb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                scenedb = cPickle.load(fid)
            print '{} gt scenedb loaded from {}'.format(self.name, cache_file)
            self.num_images = len(scenedb)
            self.scenedb = scenedb
            return scenedb

# <produce scenedb ---------------------------------------------------------------
        with open(self.database_csv_fname,'r') as fid:
            database_csv = [x for x in fid.readlines()]
        #database_csv = np.unique(database_csv).tolist()

        scene2dbi = {}

        class_name_idx = 1
        left_idx = 4
        top_idx = 5
        width_idx = 6
        height_idx = 7
        img_fname_idx = 8
        imw_idx = 9
        imh_idx = 10
        
        scenedb = []        
        for id, line in enumerate(database_csv):
            line = line.strip()
            fields = line.split(',')
            class_name = fields[class_name_idx]
            image_name = fields[img_fname_idx]
            if class_name not in self.className_to_ord.keys():
                continue
            class_ord = self.className_to_ord[class_name]
            class_idx = class_ord - 1 # used in oneHot --> 'gt_overlaps'
            left = int(fields[left_idx])
            top = int(fields[top_idx])
            width = int(fields[width_idx])
            height = int(fields[height_idx])
            img_width = int(fields[imw_idx])
            img_height = int(fields[imh_idx])

            if left <= 0:
                left = 1
            right = left + width
            if top <= 0:
                top = 1
            bottom = top + height
            if right>= img_width:
                right = img_width - 1
            if bottom>= img_height:
                bottom = img_height - 1

            gt_overlaps = np.zeros((1, self.num_classes), dtype=np.float32)
            gt_overlaps[0, class_idx] = 1
            BBs = np.expand_dims([left, top, right, bottom],axis=0)
            img_path = os.path.join(self.dataset_path, 'images', image_name)

            scene, view = get_view(image_name)
            if scene in scene2dbi:  # add to existing scene:
                dbi = scene2dbi[scene]
                scenedb[dbi]['boxes_views'][view]=np.concatenate((scenedb[dbi]['boxes_views'][view],BBs),axis=0)
                scenedb[dbi]['image_views'][view]=img_path
                scenedb[dbi]['width_views'][view] = int(img_width)
                scenedb[dbi]['height_views'][view] = int(img_height)
                scenedb[dbi]['gt_classes'][view]+=[class_ord]
                scenedb[dbi]['gt_names'][view]+=[class_name]
                scenedb[dbi]['flipped'] = False
                #scenedb[dbi]['gt_overlaps'] = np.concatenate((scenedb[dbi]['gt_overlaps'], gt_overlaps), axis=0)

            else: # new scene
                boxes_views = {'top':np.zeros((0,4)), 'left':np.zeros((0,4)), 'right':np.zeros((0,4))}
                image_views,width_views,height_views,gt_classes_views,gt_names_views =[ {'top':[], 'left':[], 'right':[]} for _ in range(5)]
                #width_views = {'top':None, 'left':None, 'right':None}
                #height_views = {'top': None, 'left': None, 'right': None}
                boxes_views[view]=BBs
                image_views[view]=img_path
                gt_classes_views[view] = [class_ord]
                gt_names_views[view] = [class_name]
                width_views[view] = int(img_width)
                height_views[view] = int(img_height)
                sc_entry = {
                    'boxes_views':boxes_views,
                    'image_views':image_views,
                    'gt_classes': gt_classes_views,
                    'gt_names': gt_names_views,
                    'width_views':width_views,
                    'height_views':height_views
                    #'gt_overlaps': gt_overlaps,
                }
                scenedb +=[sc_entry]
                scene2dbi[scene] = len(scenedb)-1


            # append = False
            # for nImg, entry in enumerate(roidb):
            #     if entry['image']==img_path:
            #         append = True
            #         append_idx = nImg
            # if append:
            #     roidb[append_idx]['boxes'] =np.concatenate((roidb[append_idx]['boxes'],BBs),axis=0)
            #     roidb[append_idx]['gt_classes']+=[class_ord]
            #     roidb[append_idx]['gt_names']+=[class_name]
            #     roidb[append_idx]['gt_overlaps'] = np.concatenate((roidb[append_idx]['gt_overlaps'],gt_overlaps), axis=0)
            #     continue
            # #im = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # roidb.append({
            #     'boxes': BBs,
            #     'flipped': False,
            #     'gt_classes': [class_ord],
            #     'gt_names':[class_name],
            #     'gt_overlaps': gt_overlaps,
            #     'width': int(img_width),
            #     'height': int(img_height),
            #     'image': img_path,
            #     #'max_classes': class_ord
            #     #'max_overlaps': np.ones((BBs.shape[0], 1), dtype=np.float32),
            #     #'aug_gen': aug_gen
            # })

        scenedb_new = []
        for entry in scenedb:
            entry['gt_classes'] = entry['gt_classes']
            entry['gt_classes'] = np.asarray(entry['gt_classes']['top'])
            entry['max_classes'] = entry['gt_classes']
            entry['max_overlaps'] = []
            scenedb_new.append(entry)


        # scenedb_new = []
        # cntr = 0
        # for ei,entry in enumerate(scenedb):
        #     #print(ei)
        #     if entry['gt_classes']['top']==entry['gt_classes']['left'] and entry['gt_classes']['top']==entry['gt_classes']['right']:
        #         entry['gt_classes'] = entry['gt_classes']['top']
        #         entry['gt_classes'] = np.asarray(entry['gt_classes'])
        #         entry['max_classes'] = entry['gt_classes']
        #         entry['max_overlaps'] = []
        #         scenedb_new.append(entry)
        #     else:
        #         print('entry #{0} - {1} discarded'.format(cntr, entry['image_views']['top']))
        #         cntr+=1
            # induce 'top' order on the 'left' and 'right' --------------------------------
            # base_gt = entry['gt_classes']['top']
            # for view in ['left','right']:
            #     gt_side = entry['gt_classes'][view]
            #     order = []
            #     for o in base_gt:
            #         order+=np.where(np.asarray(gt_side)==o)[0].tolist()
            #     entry['gt_classes'][view] = [gt_side[o] for o in order]
            #     bbox = np.zeros((0,4))
            #     for o in order:
            #         bbox = np.concatenate((bbox,np.expand_dims(entry['boxes_views'][view][o],axis=0)),axis=0)
            #     entry['boxes_views'][view] =bbox
            # entry['gt_classes'] = entry['gt_classes']['top']
            # # induce 'top' order on the 'left' and 'right' --------------------------------


        scenedb = scenedb_new
        self.num_images = len(scenedb)
        self.scenedb = scenedb
        
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self.scenedb, fid, cPickle.HIGHEST_PROTOCOL) # gt_roidb
        print 'wrote gt scenedb to {}'.format(cache_file)
        return self.scenedb

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        print 'append flipped images to roidb'
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roi_rec = copy.deepcopy(roidb[i])
            boxes_views = roi_rec['boxes_views'].copy()
            #boxes[np.where(boxes[:, 2]==roi_rec['width']),2]= roi_rec['width']-1# JS added to fix the {0 -> 65535} bug.
            for view in self.views_list:
                oldx1 = boxes_views[view][:, 0].copy()
                oldx2 = boxes_views[view][:, 2].copy()
                boxes_views[view][:, 0] = roi_rec['width_views'][view] - oldx2 - 1
                boxes_views[view][:, 2] = roi_rec['width_views'][view] - oldx1 - 1
                assert (boxes_views[view][:, 2] >= boxes_views[view][:, 0]).all()
            roi_rec['boxes_views'] = boxes_views
            roi_rec['flipped'] = True

            roidb.append(roi_rec)

        self.image_set_index *= 2
        return roidb

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

    def evaluate_detections(self, detections, display=False,
                                  display_folder='/dccstor/jsdata1/dev/RepMet/output/JES_pilot/disp_foods_main_ep8_corr_nms'):
        print('not implemented')
        return
    def evaluate_detections_roidb(self,detections,display=False,display_folder = '/dccstor/jsdata1/dev/RepMet/output/JES_pilot/disp_foods_main_ep8_corr_nms'):
    # detections: list of length #classes (+bkgnd). Each entry is a list of size #<test images>, of arrays of 5-row detections, some are empty.
        from utils.show_boxes import show_boxes, show_dets_gt_boxes
        from utils.PerfStats import PerfStats
        assert_folder(display_folder)
        ovthresh = 0.5
        score_thresh = 0.1
        Nclasses = len(detections)
        Nimages = len(self.scenedb)
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
        for idx, name in enumerate(self.class_names):
            ord = idx+1
            name2ord[name] = ord
            ord2name[ord] = name

        for img_num, entry in enumerate(self.scenedb):
            im_path, im_fname = os.path.split(entry['image'])
            dets_export_fname = os.path.join(display_folder,im_fname[:-4]+'.txt')
            #if im_fname[-8:-4]=='_top':
            self.export_dets_B2C(dets_reflow[img_num][1:],dets_export_fname,ord2name)

            stats.comp_epi_stats_m(dets_reflow[img_num][1:],entry['boxes'],entry['gt_classes'], epi_cats,ovthresh)
            if display:
                im = cv2.imread(entry['image'])
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im_path, im_fname = os.path.split(entry['image'])
                show_dets_gt_boxes(im, dets_reflow[img_num][1:], self.class_names, entry['boxes'],entry['gt_names'], scale=1.0, FS=8,LW=1.5, save_file_path=os.path.join(display_folder,'disp_{0}.png'.format(im_fname)))

        my_logger = configure_logging('/dccstor/jsdata1/dev/RepMet/data/JES_pilot/tmp_logger.log')
        stats.print_perf(my_logger, prefix='')

