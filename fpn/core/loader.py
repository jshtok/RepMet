# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Copyright (c) 2019 IBM Corp
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi
# --------------------------------------------------------

import os
import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice
import cPickle

from config.config import config
from rpn.rpn import get_rpn_testbatch, get_rpn_batch, assign_pyramid_anchor
from rcnn import get_rcnn_testbatch


def par_assign_anchor_wrapper(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label, img_fname = get_rpn_batch(iroidb, cfg)
    data_shape = {k: v.shape for k, v in data.items()}
    del data_shape['im_info']

    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]

    if not cfg.network.base_net_lock:
        feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
        label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], data['im_info'], cfg,
                                      feat_strides, anchor_scales, anchor_ratios, allowed_border)
    else:
        label = None
    return {'data': data, 'label': label,'img_fname':img_fname}

class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False,
                 has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if has_rpn:
            self.data_name = ['data', 'im_info']
        else:
            self.data_name = ['data', 'rois']
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, idata)] for idata in self.data]

    @property
    def provide_label(self):
        return [None for _ in range(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

        # self.filter_logic()

    def filter_logic(self):
        sel_set=[]
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            cats = cats[cats < (self.cfg.dataset.NUM_CLASSES-1)]
            if not cats.size:
                continue
            sel_set.append(cur)
        sel_set=np.array(sel_set)
        if self.shuffle:
            p = np.random.permutation(np.arange(len(sel_set)))
            sel_set = sel_set[p]
        self.index = sel_set
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.im_info, mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        #print(roidb[0]['image'])
        #print(roidb[0]['gt_names'])
        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb, self.cfg)
        else:
            data, label, im_info = get_rcnn_testbatch(roidb, self.cfg)
        self.data = [[mx.nd.array(idata[name]) for name in self.data_name] for idata in data]
        self.im_info = im_info

    def get_batch_individual(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb, self.cfg)
        else:
            data, label, im_info = get_rcnn_testbatch(roidb, self.cfg)
        self.data = [mx.nd.array(data[name]) for name in self.data_name]
        self.im_info = im_info

class PyramidAnchorIterator(mx.io.DataIter):

    # pool = Pool(processes=4)
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        import random
        random.seed(901)
        from random import shuffle
        self.roidb = roidb
        shuffle(self.roidb)
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        if self.cfg.network.base_net_lock:
            self.label_name = []
        else:
            self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.img_fname= None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.size = len(self.roidb)
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

        #self.apply_index_constraints()
        if self.cfg.dataset.order_classes_incrementally:
            self.order_classes_incrementally()
        if self.cfg.dataset.balance_classes:
            self.balance_classes()

    def balance_classes(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        cnts = np.zeros((10000))
        sel_set=[]
        sel_set_cats=[]

        if config.dataset.cls_filter_files is not None:
            fls = config.dataset.cls_filter_files.split(':')
            with open(fls[0],'rb') as f:
                cls2id_map = cPickle.load(f)
            with open(fls[1]) as f:
                classes2use = [x.strip().lower() for x in f.readlines()]
                #classes2use = [x.strip() for x in f.readlines()]
            clsIds2use = set()
            for cls in classes2use:
                clsIds2use.add(cls2id_map[cls])
            self.cfg.dataset.clsIds2use = clsIds2use.copy()
            self.cfg.dataset.clsIds2use.add(0)

        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            if config.dataset.cls_filter_files is not None:
                cats = np.array([x for x in cats if (x+1) in clsIds2use])
            # else:
                #     cats = cats[cats < (self.cfg.dataset.NUM_CLASSES-1)]

            if not cats.size:
                continue
            ix = np.argmin(cnts[cats])
            if cnts[cats[ix]] < num_ex_per_class:
                cnts[cats[ix]] += 1
            else:
                continue #not adding more examples, each epoch runs in random order of this
            sel_set.append(cur)
            sel_set_cats.append(cats)
        sel_set=np.array(sel_set)
        p = np.random.permutation(np.arange(len(sel_set)))
        sel_set = sel_set[p]
        self.index = sel_set
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def order_classes_incrementally(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        num_ex_between_extras = self.cfg.dataset.num_ex_between_extras
        cls=[x['gt_classes'] for x in self.roidb]
        base_set=[]
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        base_flags = np.zeros((num_classes,),dtype=bool)
        if self.cfg.dataset.num_ex_base_limit > 0:
            base_cnts = np.zeros((10000))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            is_base = True
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    is_base = False
            base_flags[cats] = is_base
            if is_base:
                if self.cfg.dataset.num_ex_base_limit > 0:
                    ix = np.argmin(base_cnts[cats])
                    if base_cnts[cats[ix]] < self.cfg.dataset.num_ex_base_limit:
                        base_cnts[cats[ix]] += 1
                    else:
                        continue #not adding more examples, each epoch runs in random order of this
                base_set.append(cur)
        base_set=np.array(base_set)
        inds=[]
        extra_cat_inds=[i for i in range(len(base_flags)) if not base_flags[i]]
        for iC, C in enumerate(extra_cat_inds):
            print(C)
            if iC > self.cfg.dataset.max_num_extra_classes:
                break
            base_set_ind = 0
            cat_ix = np.array([i for i in range(len(cls)) if C+1 in cls[i]])
            p = np.random.permutation(np.arange(len(cat_ix)))
            cat_ix = cat_ix[p]
            for iE in range(num_ex_per_class):
                inds.append(np.array([cat_ix[iE]]))
                inds.append(base_set[base_set_ind:base_set_ind+num_ex_between_extras])
                if base_set_ind >= (len(base_set)-num_ex_between_extras):
                    base_set_ind = 0
                else:
                    base_set_ind += num_ex_between_extras
            base_set = np.concatenate((base_set,cat_ix[0:num_ex_per_class]))
            p = np.random.permutation(np.arange(len(base_set)))
            base_set = base_set[p]
        inds=np.concatenate(inds)
        self.index = inds
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def apply_index_constraints(self):
        # self.roidb, per_category_epoch_max
        # self.index
        valid = np.ones(self.index.shape,dtype=bool)
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        cls_counts = np.zeros((num_classes,))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    cats = roi['gt_classes'] - 1 # minus 1 for excluding BG
                    if np.any(cls_counts[cats] < m):
                        cls_counts[cats] += 1
                    else:
                        valid[ix] = False
        self.index = self.index[valid]
        self.size = len(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]

        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # if len(roidb)>0:
        #     print('index '+str(self.index[cur_from]) )
        #     for entry in roidb:
        #         print(entry['image'])
        #         print('width '+ str(entry['width']))
        #         print('height ' + str(entry['height']))
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        all_img_fname = [_['img_fname'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        self.img_fname = all_img_fname


class PyramidAnchorIterator_resumable(mx.io.DataIter):

    # pool = Pool(processes=4)
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, order=None):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator_resumable, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        import random
        random.seed(901)
        from random import shuffle
        self.roidb = roidb
        self.order = np.random.permutation(len(roidb)) if order is None else order

        self.roidb = [self.roidb[idx] for idx in self.order]
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        if self.cfg.network.base_net_lock:
            self.label_name = []
        else:
            self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.img_fname= None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.size = len(self.roidb)
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

        #self.apply_index_constraints()
        if self.cfg.dataset.order_classes_incrementally:
            self.order_classes_incrementally()
        if self.cfg.dataset.balance_classes:
            self.balance_classes()

    def balance_classes(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        cnts = np.zeros((10000))
        sel_set=[]
        sel_set_cats=[]

        if config.dataset.cls_filter_files is not None:
            fls = config.dataset.cls_filter_files.split(':')
            with open(fls[0],'rb') as f:
                cls2id_map = cPickle.load(f)
            with open(fls[1]) as f:
                classes2use = [x.strip().lower() for x in f.readlines()]
                #classes2use = [x.strip() for x in f.readlines()]
            clsIds2use = set()
            for cls in classes2use:
                clsIds2use.add(cls2id_map[cls])
            self.cfg.dataset.clsIds2use = clsIds2use.copy()
            self.cfg.dataset.clsIds2use.add(0)

        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            if config.dataset.cls_filter_files is not None:
                cats = np.array([x for x in cats if (x+1) in clsIds2use])
            # else:
                #     cats = cats[cats < (self.cfg.dataset.NUM_CLASSES-1)]

            if not cats.size:
                continue
            ix = np.argmin(cnts[cats])
            if cnts[cats[ix]] < num_ex_per_class:
                cnts[cats[ix]] += 1
            else:
                continue #not adding more examples, each epoch runs in random order of this
            sel_set.append(cur)
            sel_set_cats.append(cats)
        sel_set=np.array(sel_set)
        p = np.random.permutation(np.arange(len(sel_set)))
        sel_set = sel_set[p]
        self.index = sel_set
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def order_classes_incrementally(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        num_ex_between_extras = self.cfg.dataset.num_ex_between_extras
        cls=[x['gt_classes'] for x in self.roidb]
        base_set=[]
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        base_flags = np.zeros((num_classes,),dtype=bool)
        if self.cfg.dataset.num_ex_base_limit > 0:
            base_cnts = np.zeros((10000))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            is_base = True
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    is_base = False
            base_flags[cats] = is_base
            if is_base:
                if self.cfg.dataset.num_ex_base_limit > 0:
                    ix = np.argmin(base_cnts[cats])
                    if base_cnts[cats[ix]] < self.cfg.dataset.num_ex_base_limit:
                        base_cnts[cats[ix]] += 1
                    else:
                        continue #not adding more examples, each epoch runs in random order of this
                base_set.append(cur)
        base_set=np.array(base_set)
        inds=[]
        extra_cat_inds=[i for i in range(len(base_flags)) if not base_flags[i]]
        for iC, C in enumerate(extra_cat_inds):
            print(C)
            if iC > self.cfg.dataset.max_num_extra_classes:
                break
            base_set_ind = 0
            cat_ix = np.array([i for i in range(len(cls)) if C+1 in cls[i]])
            p = np.random.permutation(np.arange(len(cat_ix)))
            cat_ix = cat_ix[p]
            for iE in range(num_ex_per_class):
                inds.append(np.array([cat_ix[iE]]))
                inds.append(base_set[base_set_ind:base_set_ind+num_ex_between_extras])
                if base_set_ind >= (len(base_set)-num_ex_between_extras):
                    base_set_ind = 0
                else:
                    base_set_ind += num_ex_between_extras
            base_set = np.concatenate((base_set,cat_ix[0:num_ex_per_class]))
            p = np.random.permutation(np.arange(len(base_set)))
            base_set = base_set[p]
        inds=np.concatenate(inds)
        self.index = inds
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def apply_index_constraints(self):
        # self.roidb, per_category_epoch_max
        # self.index
        valid = np.ones(self.index.shape,dtype=bool)
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        cls_counts = np.zeros((num_classes,))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    cats = roi['gt_classes'] - 1 # minus 1 for excluding BG
                    if np.any(cls_counts[cats] < m):
                        cls_counts[cats] += 1
                    else:
                        valid[ix] = False
        self.index = self.index[valid]
        self.size = len(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]

        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # if len(roidb)>0:
        #     print('index '+str(self.index[cur_from]) )
        #     for entry in roidb:
        #         print(entry['image'])
        #         print('width '+ str(entry['width']))
        #         print('height ' + str(entry['height']))
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        all_img_fname = [_['img_fname'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        self.img_fname = all_img_fname



#-----------------------------------------------------------------------------------
#      PFP
#-----------------------------------------------------------------------------------

def par_assign_anchor_wrapper_pre_1(cfg, iroidb, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label, img_fname = get_rpn_batch(iroidb, cfg)
    _, im_fname = os.path.split(img_fname)
    rps_fname = os.path.join(cfg.work_root, 'precomputed_data', im_fname.replace('.jpg', '_feat.pkl'))
    with open(rps_fname,'rb') as fid:
        state_data = cPickle.load(fid)

    fc_new_1_relu_pre = state_data['fc_new_1_relu_pre']
    label = state_data['label_pre']
    rois_pre = state_data['rois_pre']
    bbox_weight_pre = state_data['bbox_weight_pre']
    bbox_target_pre = state_data['bbox_target_pre']

    del data['data']
    data['fc_new_1_relu_pre'] = fc_new_1_relu_pre
    del data['im_info']

    data_shape = {k: v.shape for k, v in data.items()}


    # add gt_boxes to data for e2e
    #data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]
    data['rois_pre'] = rois_pre
    data['bbox_weight_pre'] = bbox_weight_pre
    data['bbox_target_pre'] = bbox_target_pre
    # if not cfg.network.base_net_lock:
    #     feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
    #     label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], data['im_info'], cfg,
    #                                   feat_strides, anchor_scales, anchor_ratios, allowed_border)
    # else:
    #label = None
    return {'data': data, 'label': label,'img_fname':img_fname}

def par_assign_anchor_wrapper_pre_2(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border,data_names):
    # get testing data for multigpu
    data, rpn_label, img_fname = get_rpn_batch(iroidb, cfg)
    # data_shape = {k: v.shape for k, v in data.items()}
    del data['data']
    #del data['im_info']

    _, im_fname = os.path.split(img_fname)
    rps_fname = os.path.join(cfg.work_root, 'precomputed_data', im_fname.replace('.jpg', '_feat.pkl'))
    with open(rps_fname,'rb') as fid:
        state_data = cPickle.load(fid)
    data['fpn_p2_pre'] = state_data['fpn_p2_pre']
    data['fpn_p3_pre'] = state_data['fpn_p3_pre']
    data['fpn_p4_pre'] = state_data['fpn_p4_pre']
    data['fpn_p5_pre'] = state_data['fpn_p5_pre']
    data['fpn_p6_pre'] = state_data['fpn_p6_pre']
    data['label_pre'] = state_data['label_pre']
    #data_names = ['fpn_p2_pre','fpn_p3_pre','fpn_p4_pre','fpn_p5_pre','fpn_p6_pre']

    data_shape = {k: v.shape for k, v in data.items()}

    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]

    if not cfg.network.base_net_lock:

        # feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
        feat_shape = [y[1] for y in [x.infer_shape(**{k : data[k].shape}) for x, k in zip(feat_sym,data_names)]]

        label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], data['im_info'], cfg,
                                      feat_strides, anchor_scales, anchor_ratios, allowed_border)
    else:
        label = None
    return {'data': data, 'label': label,'img_fname':img_fname}

class PyramidAnchorIterator_pre_1(mx.io.DataIter):

    # pool = Pool(processes=4)
    def __init__(self, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator_pre_1, self).__init__()

        # save parameters as properties
        #self.feat_sym = feat_sym
        import random
        random.seed(901)
        from random import shuffle
        self.roidb = roidb
        shuffle(self.roidb)
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['fc_new_1_relu_pre',  'rois_pre','bbox_weight_pre','bbox_target_pre'] # 'gt_boxes',
        else:
            self.data_name = ['fc_new_1_relu_pre']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        if self.cfg.network.base_net_lock:
            self.label_name = []
        else:
            self.label_name = ['label_pre', 'bbox_target_pre', 'bbox_weight_pre']
            #self.label_name = ['label_pre']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.img_fname= None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[0])]]
        #return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.label))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.size = len(self.roidb)
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

        #self.apply_index_constraints()
        if self.cfg.dataset.order_classes_incrementally:
            self.order_classes_incrementally()
        if self.cfg.dataset.balance_classes:
            self.balance_classes()

    def balance_classes(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        cnts = np.zeros((10000))
        sel_set=[]
        sel_set_cats=[]

        if config.dataset.cls_filter_files is not None:
            import cPickle
            fls = config.dataset.cls_filter_files.split(':')
            with open(fls[0],'rb') as f:
                cls2id_map = cPickle.load(f)
            with open(fls[1]) as f:
                classes2use = [x.strip().lower() for x in f.readlines()]
                #classes2use = [x.strip() for x in f.readlines()]
            clsIds2use = set()
            for cls in classes2use:
                clsIds2use.add(cls2id_map[cls])
            self.cfg.dataset.clsIds2use = clsIds2use.copy()
            self.cfg.dataset.clsIds2use.add(0)

        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            if config.dataset.cls_filter_files is None:
                cats = cats[cats < (self.cfg.dataset.NUM_CLASSES-1)]
            else:
                cats = np.array([x for x in cats if (x+1) in clsIds2use])
            if not cats.size:
                continue
            ix = np.argmin(cnts[cats])
            if cnts[cats[ix]] < num_ex_per_class:
                cnts[cats[ix]] += 1
            else:
                continue #not adding more examples, each epoch runs in random order of this
            sel_set.append(cur)
            sel_set_cats.append(cats)
        sel_set=np.array(sel_set)
        p = np.random.permutation(np.arange(len(sel_set)))
        sel_set = sel_set[p]
        self.index = sel_set
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def order_classes_incrementally(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        num_ex_between_extras = self.cfg.dataset.num_ex_between_extras
        cls=[x['gt_classes'] for x in self.roidb]
        base_set=[]
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        base_flags = np.zeros((num_classes,),dtype=bool)
        if self.cfg.dataset.num_ex_base_limit > 0:
            base_cnts = np.zeros((10000))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            is_base = True
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    is_base = False
            base_flags[cats] = is_base
            if is_base:
                if self.cfg.dataset.num_ex_base_limit > 0:
                    ix = np.argmin(base_cnts[cats])
                    if base_cnts[cats[ix]] < self.cfg.dataset.num_ex_base_limit:
                        base_cnts[cats[ix]] += 1
                    else:
                        continue #not adding more examples, each epoch runs in random order of this
                base_set.append(cur)
        base_set=np.array(base_set)
        inds=[]
        extra_cat_inds=[i for i in range(len(base_flags)) if not base_flags[i]]
        for iC, C in enumerate(extra_cat_inds):
            print(C)
            if iC > self.cfg.dataset.max_num_extra_classes:
                break
            base_set_ind = 0
            cat_ix = np.array([i for i in range(len(cls)) if C+1 in cls[i]])
            p = np.random.permutation(np.arange(len(cat_ix)))
            cat_ix = cat_ix[p]
            for iE in range(num_ex_per_class):
                inds.append(np.array([cat_ix[iE]]))
                inds.append(base_set[base_set_ind:base_set_ind+num_ex_between_extras])
                if base_set_ind >= (len(base_set)-num_ex_between_extras):
                    base_set_ind = 0
                else:
                    base_set_ind += num_ex_between_extras
            base_set = np.concatenate((base_set,cat_ix[0:num_ex_per_class]))
            p = np.random.permutation(np.arange(len(base_set)))
            base_set = base_set[p]
        inds=np.concatenate(inds)
        self.index = inds
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def apply_index_constraints(self):
        # self.roidb, per_category_epoch_max
        # self.index
        valid = np.ones(self.index.shape,dtype=bool)
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        cls_counts = np.zeros((num_classes,))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    cats = roi['gt_classes'] - 1 # minus 1 for excluding BG
                    if np.any(cls_counts[cats] < m):
                        cls_counts[cats] += 1
                    else:
                        valid[ix] = False
        self.index = self.index[valid]
        self.size = len(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        feat_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        return max_data_shape, max_label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # if len(roidb)>0:
        #     print('index '+str(self.index[cur_from]) )
        #     for entry in roidb:
        #         print(entry['image'])
        #         print('width '+ str(entry['width']))
        #         print('height ' + str(entry['height']))
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper_pre_1(self.cfg, iroidb, self.feat_strides, self.anchor_scales,
                                                       self.anchor_ratios, self.allowed_border))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        all_img_fname = [_['img_fname'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label)] for label in all_label]
        self.img_fname = all_img_fname

class PyramidAnchorIterator_pre_2(mx.io.DataIter):

    # pool = Pool(processes=4)
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator_pre_2, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        import random
        random.seed(901)
        from random import shuffle
        self.roidb = roidb
        shuffle(self.roidb)
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['fpn_p2_pre','fpn_p3_pre','fpn_p4_pre','fpn_p5_pre','fpn_p6_pre','im_info','gt_boxes']
        else:
            self.data_name = ['data']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        if self.cfg.network.base_net_lock:
            self.label_name = []
        else:
            self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.img_fname= None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.size = len(self.roidb)
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

        #self.apply_index_constraints()
        if self.cfg.dataset.order_classes_incrementally:
            self.order_classes_incrementally()
        if self.cfg.dataset.balance_classes:
            self.balance_classes()

    def balance_classes(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        cnts = np.zeros((10000))
        sel_set=[]
        sel_set_cats=[]

        if config.dataset.cls_filter_files is not None:
            fls = config.dataset.cls_filter_files.split(':')
            with open(fls[0],'rb') as f:
                cls2id_map = cPickle.load(f)
            with open(fls[1]) as f:
                classes2use = [x.strip().lower() for x in f.readlines()]
                #classes2use = [x.strip() for x in f.readlines()]
            clsIds2use = set()
            for cls in classes2use:
                clsIds2use.add(cls2id_map[cls])
            self.cfg.dataset.clsIds2use = clsIds2use.copy()
            self.cfg.dataset.clsIds2use.add(0)

        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            if config.dataset.cls_filter_files is not None:
            #     cats = cats[cats < (self.cfg.dataset.NUM_CLASSES-1)]
            # else:
                cats = np.array([x for x in cats if (x+1) in clsIds2use])
            if not cats.size:
                continue
            ix = np.argmin(cnts[cats])
            if cnts[cats[ix]] < num_ex_per_class:
                cnts[cats[ix]] += 1
            else:
                continue #not adding more examples, each epoch runs in random order of this
            sel_set.append(cur)
            sel_set_cats.append(cats)
        sel_set=np.array(sel_set)
        p = np.random.permutation(np.arange(len(sel_set)))
        sel_set = sel_set[p]
        self.index = sel_set
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def order_classes_incrementally(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        num_ex_between_extras = self.cfg.dataset.num_ex_between_extras
        cls=[x['gt_classes'] for x in self.roidb]
        base_set=[]
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        base_flags = np.zeros((num_classes,),dtype=bool)
        if self.cfg.dataset.num_ex_base_limit > 0:
            base_cnts = np.zeros((10000))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            is_base = True
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    is_base = False
            base_flags[cats] = is_base
            if is_base:
                if self.cfg.dataset.num_ex_base_limit > 0:
                    ix = np.argmin(base_cnts[cats])
                    if base_cnts[cats[ix]] < self.cfg.dataset.num_ex_base_limit:
                        base_cnts[cats[ix]] += 1
                    else:
                        continue #not adding more examples, each epoch runs in random order of this
                base_set.append(cur)
        base_set=np.array(base_set)
        inds=[]
        extra_cat_inds=[i for i in range(len(base_flags)) if not base_flags[i]]
        for iC, C in enumerate(extra_cat_inds):
            print(C)
            if iC > self.cfg.dataset.max_num_extra_classes:
                break
            base_set_ind = 0
            cat_ix = np.array([i for i in range(len(cls)) if C+1 in cls[i]])
            p = np.random.permutation(np.arange(len(cat_ix)))
            cat_ix = cat_ix[p]
            for iE in range(num_ex_per_class):
                inds.append(np.array([cat_ix[iE]]))
                inds.append(base_set[base_set_ind:base_set_ind+num_ex_between_extras])
                if base_set_ind >= (len(base_set)-num_ex_between_extras):
                    base_set_ind = 0
                else:
                    base_set_ind += num_ex_between_extras
            base_set = np.concatenate((base_set,cat_ix[0:num_ex_per_class]))
            p = np.random.permutation(np.arange(len(base_set)))
            base_set = base_set[p]
        inds=np.concatenate(inds)
        self.index = inds
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def apply_index_constraints(self):
        # self.roidb, per_category_epoch_max
        # self.index
        valid = np.ones(self.index.shape,dtype=bool)
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        cls_counts = np.zeros((num_classes,))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    cats = roi['gt_classes'] - 1 # minus 1 for excluding BG
                    if np.any(cls_counts[cats] < m):
                        cls_counts[cats] += 1
                    else:
                        valid[ix] = False
        self.index = self.index[valid]
        self.size = len(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        feat_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        return max_data_shape, max_label_shape

    # def infer_shape(self, max_data_shape=None, max_label_shape=None):
    #     """ Return maximum data and label shape for single gpu """
    #     if max_data_shape is None:
    #         max_data_shape = []
    #     if max_label_shape is None:
    #         max_label_shape = []
    #     max_shapes = dict(max_data_shape + max_label_shape)
    #     #input_batch_size = max_shapes['data'][0]
    #     #im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
    #
    #     feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
    #     label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
    #                                   self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
    #     label = [label[k] for k in self.label_name]
    #     label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
    #
    #     return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # if len(roidb)>0:
        #     print('index '+str(self.index[cur_from]) )
        #     for entry in roidb:
        #         print(entry['image'])
        #         print('width '+ str(entry['width']))
        #         print('height ' + str(entry['height']))
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper_pre_2(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border,self.data_name))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        all_img_fname = [_['img_fname'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        self.img_fname = all_img_fname

#-----------------------------------------------------------------------------------
#      scene
#-----------------------------------------------------------------------------------

from rpn.rpn import get_rpn_batch_scene,get_rpn_batch_scene2

class PyramidAnchorIterator_scene(mx.io.DataIter):

    # pool = Pool(processes=4)
    def __init__(self, feat_sym, scenedb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param scenedb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects. Not implemented.
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator_scene, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        import random
        random.seed(901)
        from random import shuffle
        self.scenedb = scenedb
        shuffle(self.scenedb)
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border


        # infer properties from scenedb
        self.size = len(scenedb)
        self.index = np.arange(self.size)

        views_list = scenedb[0]['boxes_views'].keys()
        if False: # 3D tfm lab
            calib_file ='/dccstor/jsdata1/dev/RepMet/data/JES_pilot/cam_setup.txt'
            from utils.JES3D_transform import JES3D_transform
            Htfm = JES3D_transform(calib_file)
            box_top = scenedb[0]['boxes_views']['top'][4]
            pt_top = [box_top[0], box_top[1]]
            pt_left = Htfm.trans_rot(pt_top, 2,0)
            print(pt_left)
            pt_right = Htfm.trans_rot(pt_top, 2, 1)
            print(pt_right)
            pt_top = [box_top[2], box_top[3]]
            pt_left = Htfm.trans_rot(pt_top, 2,0)
            print(pt_left)
            pt_right = Htfm.trans_rot(pt_top, 2, 1)
            print(pt_right)
            a = 1
        # decide data and label names
        self.data_name =[]
        if self.cfg.TRAIN.END2END:
            #for view in views_list:
                #self.data_name.append('data_' + view)
                #self.data_name.append('im_info_' + view)
            self.data_name.append('data')
            self.data_name.append('im_info_top')
            self.data_name.append('gt_boxes')
            #self.data_name.append('homog_data')
        else:
            self.data_name.append('data')
           # self.data_name.append(['data_' + view])

        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        self.label_name = []
        if not self.cfg.network.base_net_lock:
            # for view in views_list:
            #     self.label_name.append('label_' + view, 'bbox_target_' + view, 'bbox_weight_' + view])
            self.label_name.extend(['label', 'bbox_target', 'bbox_weight'])
        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.img_fname= None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.size = len(self.scenedb)
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

        #self.apply_index_constraints()
        if self.cfg.dataset.order_classes_incrementally:
            self.order_classes_incrementally()
        if self.cfg.dataset.balance_classes:
            self.balance_classes()

    def balance_classes(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        cnts = np.zeros((10000))
        sel_set=[]
        sel_set_cats=[]

        if config.dataset.cls_filter_files is not None:
            fls = config.dataset.cls_filter_files.split(':')
            with open(fls[0],'rb') as f:
                cls2id_map = cPickle.load(f)
            with open(fls[1]) as f:
                classes2use = [x.strip().lower() for x in f.readlines()]
                #classes2use = [x.strip() for x in f.readlines()]
            clsIds2use = set()
            for cls in classes2use:
                clsIds2use.add(cls2id_map[cls])
            self.cfg.dataset.clsIds2use = clsIds2use.copy()
            self.cfg.dataset.clsIds2use.add(0)

        for ix, cur in enumerate(self.index):
            roi = self.scenedb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            if config.dataset.cls_filter_files is not None:
                cats = np.array([x for x in cats if (x+1) in clsIds2use])
            # else:
                #     cats = cats[cats < (self.cfg.dataset.NUM_CLASSES-1)]

            if not cats.size:
                continue
            ix = np.argmin(cnts[cats])
            if cnts[cats[ix]] < num_ex_per_class:
                cnts[cats[ix]] += 1
            else:
                continue #not adding more examples, each epoch runs in random order of this
            sel_set.append(cur)
            sel_set_cats.append(cats)
        sel_set=np.array(sel_set)
        p = np.random.permutation(np.arange(len(sel_set)))
        sel_set = sel_set[p]
        self.index = sel_set
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def order_classes_incrementally(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        num_ex_between_extras = self.cfg.dataset.num_ex_between_extras
        cls=[x['gt_classes'] for x in self.scenedb]
        base_set=[]
        num_classes = np.max([np.max(x['gt_classes']) for x in self.scenedb])
        base_flags = np.zeros((num_classes,),dtype=bool)
        if self.cfg.dataset.num_ex_base_limit > 0:
            base_cnts = np.zeros((10000))
        for ix, cur in enumerate(self.index):
            roi = self.scenedb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            is_base = True
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    is_base = False
            base_flags[cats] = is_base
            if is_base:
                if self.cfg.dataset.num_ex_base_limit > 0:
                    ix = np.argmin(base_cnts[cats])
                    if base_cnts[cats[ix]] < self.cfg.dataset.num_ex_base_limit:
                        base_cnts[cats[ix]] += 1
                    else:
                        continue #not adding more examples, each epoch runs in random order of this
                base_set.append(cur)
        base_set=np.array(base_set)
        inds=[]
        extra_cat_inds=[i for i in range(len(base_flags)) if not base_flags[i]]
        for iC, C in enumerate(extra_cat_inds):
            print(C)
            if iC > self.cfg.dataset.max_num_extra_classes:
                break
            base_set_ind = 0
            cat_ix = np.array([i for i in range(len(cls)) if C+1 in cls[i]])
            p = np.random.permutation(np.arange(len(cat_ix)))
            cat_ix = cat_ix[p]
            for iE in range(num_ex_per_class):
                inds.append(np.array([cat_ix[iE]]))
                inds.append(base_set[base_set_ind:base_set_ind+num_ex_between_extras])
                if base_set_ind >= (len(base_set)-num_ex_between_extras):
                    base_set_ind = 0
                else:
                    base_set_ind += num_ex_between_extras
            base_set = np.concatenate((base_set,cat_ix[0:num_ex_per_class]))
            p = np.random.permutation(np.arange(len(base_set)))
            base_set = base_set[p]
        inds=np.concatenate(inds)
        self.index = inds
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def apply_index_constraints(self):
        # self.scenedb, per_category_epoch_max
        # self.index
        valid = np.ones(self.index.shape,dtype=bool)
        num_classes = np.max([np.max(x['gt_classes']) for x in self.scenedb])
        cls_counts = np.zeros((num_classes,))
        for ix, cur in enumerate(self.index):
            roi = self.scenedb[cur]
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    cats = roi['gt_classes'] - 1 # minus 1 for excluding BG
                    if np.any(cls_counts[cats] < m):
                        cls_counts[cats] += 1
                    else:
                        valid[ix] = False
        self.index = self.index[valid]
        self.size = len(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]

        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        scenedb = [self.scenedb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iscenedb = [scenedb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper_scene(self.cfg, iscenedb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        all_img_fname = [_['img_fname'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        self.img_fname = all_img_fname

class PyramidAnchorIterator_scene2(mx.io.DataIter):

    # pool = Pool(processes=4)
    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_strides=(4, 8, 16, 32, 64), anchor_scales=(8, ), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(PyramidAnchorIterator_scene2, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        import random
        random.seed(901)
        from random import shuffle
        self.roidb = roidb
        shuffle(self.roidb)
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_strides = feat_strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if self.cfg.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.feat_pyramid_level = np.log2(self.cfg.network.RPN_FEAT_STRIDE).astype(int)
        # self.label_name = ['label_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_target_p' + str(x) for x in self.feat_pyramid_level] +\
        #                   ['bbox_weight_p' + str(x) for x in self.feat_pyramid_level]

        if self.cfg.network.base_net_lock:
            self.label_name = []
        else:
            self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None
        self.img_fname= None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_parallel()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.size = len(self.roidb)
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

        #self.apply_index_constraints()
        if self.cfg.dataset.order_classes_incrementally:
            self.order_classes_incrementally()
        if self.cfg.dataset.balance_classes:
            self.balance_classes()

    def balance_classes(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        cnts = np.zeros((10000))
        sel_set=[]
        sel_set_cats=[]

        if config.dataset.cls_filter_files is not None:
            fls = config.dataset.cls_filter_files.split(':')
            with open(fls[0],'rb') as f:
                cls2id_map = cPickle.load(f)
            with open(fls[1]) as f:
                classes2use = [x.strip().lower() for x in f.readlines()]
                #classes2use = [x.strip() for x in f.readlines()]
            clsIds2use = set()
            for cls in classes2use:
                clsIds2use.add(cls2id_map[cls])
            self.cfg.dataset.clsIds2use = clsIds2use.copy()
            self.cfg.dataset.clsIds2use.add(0)

        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            if config.dataset.cls_filter_files is not None:
                cats = np.array([x for x in cats if (x+1) in clsIds2use])
            # else:
                #     cats = cats[cats < (self.cfg.dataset.NUM_CLASSES-1)]

            if not cats.size:
                continue
            ix = np.argmin(cnts[cats])
            if cnts[cats[ix]] < num_ex_per_class:
                cnts[cats[ix]] += 1
            else:
                continue #not adding more examples, each epoch runs in random order of this
            sel_set.append(cur)
            sel_set_cats.append(cats)
        sel_set=np.array(sel_set)
        p = np.random.permutation(np.arange(len(sel_set)))
        sel_set = sel_set[p]
        self.index = sel_set
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def order_classes_incrementally(self):
        num_ex_per_class = self.cfg.dataset.num_ex_per_class
        num_ex_between_extras = self.cfg.dataset.num_ex_between_extras
        cls=[x['gt_classes'] for x in self.roidb]
        base_set=[]
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        base_flags = np.zeros((num_classes,),dtype=bool)
        if self.cfg.dataset.num_ex_base_limit > 0:
            base_cnts = np.zeros((10000))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            cats = roi['gt_classes'] - 1  # minus 1 for excluding BG
            is_base = True
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    is_base = False
            base_flags[cats] = is_base
            if is_base:
                if self.cfg.dataset.num_ex_base_limit > 0:
                    ix = np.argmin(base_cnts[cats])
                    if base_cnts[cats[ix]] < self.cfg.dataset.num_ex_base_limit:
                        base_cnts[cats[ix]] += 1
                    else:
                        continue #not adding more examples, each epoch runs in random order of this
                base_set.append(cur)
        base_set=np.array(base_set)
        inds=[]
        extra_cat_inds=[i for i in range(len(base_flags)) if not base_flags[i]]
        for iC, C in enumerate(extra_cat_inds):
            print(C)
            if iC > self.cfg.dataset.max_num_extra_classes:
                break
            base_set_ind = 0
            cat_ix = np.array([i for i in range(len(cls)) if C+1 in cls[i]])
            p = np.random.permutation(np.arange(len(cat_ix)))
            cat_ix = cat_ix[p]
            for iE in range(num_ex_per_class):
                inds.append(np.array([cat_ix[iE]]))
                inds.append(base_set[base_set_ind:base_set_ind+num_ex_between_extras])
                if base_set_ind >= (len(base_set)-num_ex_between_extras):
                    base_set_ind = 0
                else:
                    base_set_ind += num_ex_between_extras
            base_set = np.concatenate((base_set,cat_ix[0:num_ex_per_class]))
            p = np.random.permutation(np.arange(len(base_set)))
            base_set = base_set[p]
        inds=np.concatenate(inds)
        self.index = inds
        self.size = len(self.index)
        print('total size {0}'.format(self.size))

    def apply_index_constraints(self):
        # self.roidb, per_category_epoch_max
        # self.index
        valid = np.ones(self.index.shape,dtype=bool)
        num_classes = np.max([np.max(x['gt_classes']) for x in self.roidb])
        cls_counts = np.zeros((num_classes,))
        for ix, cur in enumerate(self.index):
            roi = self.roidb[cur]
            if 'per_category_epoch_max' in roi:
                m = float(roi['per_category_epoch_max'])
                if m>0: # zero means disabled
                    cats = roi['gt_classes'] - 1 # minus 1 for excluding BG
                    if np.any(cls_counts[cats] < m):
                        cls_counts[cats] += 1
                    else:
                        valid[ix] = False
        self.index = self.index[valid]
        self.size = len(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_parallel()
            # self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]

        feat_shape = [y[1] for y in [x.infer_shape(**max_shapes) for x in self.feat_sym]]
        label = assign_pyramid_anchor(feat_shape, np.zeros((0, 5)), im_info, self.cfg,
                                      self.feat_strides, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]

        return max_data_shape, label_shape

    def get_batch_parallel(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # if len(roidb)>0:
        #     print('index '+str(self.index[cur_from]) )
        #     for entry in roidb:
        #         print(entry['image'])
        #         print('width '+ str(entry['width']))
        #         print('height ' + str(entry['height']))
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(par_assign_anchor_wrapper_scene2(self.cfg, iroidb, self.feat_sym, self.feat_strides, self.anchor_scales,
                                                 self.anchor_ratios, self.allowed_border))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        all_img_fname = [_['img_fname'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]
        self.img_fname = all_img_fname

def par_assign_anchor_wrapper_scene2(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label, img_fname = get_rpn_batch_scene2(iroidb, cfg)
    data_shape = {k: v.shape for k, v in data.items()}
    del data_shape['im_info']

    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]

    if not cfg.network.base_net_lock:
        feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
        label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], data['im_info'], cfg,
                                      feat_strides, anchor_scales, anchor_ratios, allowed_border)
    else:
        label = None
    return {'data': data, 'label': label,'img_fname':img_fname}

def par_assign_anchor_wrapper_scene(cfg, iroidb, feat_sym, feat_strides, anchor_scales, anchor_ratios, allowed_border):
    # get testing data for multigpu
    data, rpn_label, img_fname = get_rpn_batch_scene(iroidb, cfg)
    data_shape = {k: v.shape for k, v in data.items()}
    views_list = iroidb[0]['image_views'].keys()
    for view in views_list:
        del data_shape['im_info_'+view]
    # del data_shape['homog_data']
    # add gt_boxes to data for e2e
    data['gt_boxes'] = rpn_label['gt_boxes'][np.newaxis, :, :]
    # del data_shape['data_left']
    # del data_shape['data_right']
    #data_shape['data'] = data_shape['data_top']
    #del data_shape['data_top']
    if not cfg.network.base_net_lock:
        feat_shape = [y[1] for y in [x.infer_shape(**data_shape) for x in feat_sym]]
        label = assign_pyramid_anchor(feat_shape, rpn_label['gt_boxes'], [data['im_info_top'][0]], cfg,
                                      feat_strides, anchor_scales, anchor_ratios, allowed_border)
    else:
        label = None
    return {'data': data, 'label': label,'img_fname':img_fname}
