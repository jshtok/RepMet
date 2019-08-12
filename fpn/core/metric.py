# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

import mxnet as mx
import numpy as np

import time

LEONID_METRIC_PROFILING_ENABLED = False

def get_rpn_names(cfg = None):
    if (cfg is not None) and cfg.network.base_net_lock:
        pred = []
        label = []
    else:
        pred = ['rpn_cls_prob', 'rpn_bbox_loss']
        label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if cfg.TRAIN.ENABLE_OHEM or cfg.TRAIN.END2END:
        pred.append('rcnn_label')

    ################################################
    ### added / updated by Leonid to support oneshot
    ################################################
    if cfg.network.EMBEDDING_DIM != 0:
        pred += ['representatives']

        if cfg.network.EMBED_LOSS_ENABLED:
            pred += ['embed_loss']

        if cfg.network.BG_REPS:
            pred += ['BG_model_loss']

        if cfg.network.REPS_CLS_LOSS:
            pred += ['reps_cls_loss']

        if cfg.network.ADDITIONAL_LINEAR_CLS_LOSS:
            pred += ['rcnn_cls_prob_lin']

        if cfg.network.VAL_FILTER_REGRESS:
            pred += ['val_reg_loss']

        if cfg.network.SCORE_HIST_REGRESS:
            pred += ['score_hist_loss']
            pred += ['reg_hist']


    if cfg.TRAIN.END2END:
        rpn_pred, rpn_label = get_rpn_names(cfg)
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label

def profile_on():
    if LEONID_METRIC_PROFILING_ENABLED:
        return time.time()
    else:
        return 0

def profile_off(tic, str):
    if LEONID_METRIC_PROFILING_ENABLED:
        toc = time.time()
        print(str+' {0}'.format(toc - tic))

class RCNNFGAccuracy(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNFGAccuracy, self).__init__('R-CNN FG Accuracy')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]
        num_classes = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, num_classes).argmax(axis=1).astype('int32')
        # selection of ground truth label is different from softmax or sigmoid classifier
        label = label.asnumpy().reshape(-1, ).astype('int32')
        keep_inds = np.where(label > 0)
        # filter out -1 label because of OHEM or invalid samples
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(np.equal(pred_label.flat, label.flat))
        self.num_inst += pred_label.shape[0]

        profile_off(tic, 'RCNNFGAccuracy metric')


class RPNFGFraction(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RPNFGFraction, self).__init__('Proposal FG Fraction')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]
        num_classes = pred.shape[-1]
        # selection of ground truth label is different from softmax or sigmoid classifier
        label = label.asnumpy().reshape(-1, ).astype('int32')
        fg_inds = np.where(label > 0)[0]
        bg_inds = np.where(label == 0)[0]
        self.sum_metric += fg_inds.shape[0]
        self.num_inst += (fg_inds.shape[0] + bg_inds.shape[0])

        profile_off(tic, 'RPNFGFraction metric')


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        tic = profile_on()

        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

        profile_off(tic, 'RPNAccMetric metric')

class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

        profile_off(tic, 'RCNNAccMetric metric')

class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        tic = profile_on()

        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

        profile_off(tic, 'RPNLogLossMetric metric')

class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

        profile_off(tic, 'RCNNLogLossMetric metric')

class RCNNLinLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNLinLogLossMetric, self).__init__('RCNNLinLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        pred = preds[self.pred.index('rcnn_cls_prob_lin')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

        profile_off(tic, 'RCNNLinLogLossMetric metric')

class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        tic = profile_on()

        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

        profile_off(tic, 'RPNL1LossMetric metric')


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

        profile_off(tic, 'RCNNL1LossMetric metric')

########################################
### added by Leonid for one-shot support
########################################
class RepresentativesMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, final_output_path):
        name = 'RepresentativesStats'
        super(RepresentativesMetric, self).__init__(name)
        self.pred, self.label = get_rcnn_names(cfg)
        self.prev=None
        self.iter=0
        self.compute_freq=5000
        self.final_output_path=final_output_path

    def update(self, labels, preds):
        tic = profile_on()

        if False:
            # shape = (EMBEDDING_DIM,REPS_PER_CLASS,num_classes-1)
            representatives = preds[self.pred.index('representatives')].asnumpy()

            if self.prev is not None:
                self.sum_metric += np.sum(np.abs(np.reshape(representatives-self.prev,(-1,))))
            else:
                self.sum_metric += 0
            self.prev=representatives

            # calculate num_inst (average on those kept anchors)
            num_inst = representatives.shape[0]

            self.num_inst += num_inst
        else:
            self.sum_metric += 0
            self.num_inst += 1

        # draw and store t-SNE embedding of the representatives
        if np.mod(self.iter,self.compute_freq)==(self.compute_freq-1):
            # shape = (EMBEDDING_DIM,REPS_PER_CLASS,num_classes-1)
            representatives = preds[self.pred.index('representatives')].asnumpy()

            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import os
            import time

            tic=time.clock()

            rs = list(representatives.shape)
            # num_classes = rs[2]
            # cmap = plt.get_cmap('jet',lut=num_classes)

            # cls = np.array([[iCls]*rs[1] for iCls in range(rs[2])])
            # cls = np.reshape(cls,(-1,))
            cls = np.array([np.arange(rs[2])] * rs[1])
            cls = np.reshape(cls, (-1,))
            # print(cls)

            X=representatives
            X = np.reshape(X,(X.shape[0],-1))
            X = np.transpose(X)
            X_embedded = TSNE(n_components=2,random_state=0).fit_transform(X)

            # draw and store the image
            fig, ax = plt.subplots(1, 1)
            ax.scatter(X_embedded[:,0],
                       X_embedded[:,1],
                       c=cls,
                       s=1)
                       # cmap=cmap)
            ax.axis('off')

            fig.savefig(os.path.join(self.final_output_path,'latest_rep_tsne.png'))
            plt.close(fig)

            toc = time.clock()
            # print('Representatives display elapsed time: {0} sec'.format(toc-tic))

        # increase the iter counter
        self.iter += 1

        profile_off(tic, 'RepresentativesMetric metric')

class EmbedMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        name = 'EmbedLoss'
        super(EmbedMetric, self).__init__(name)
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        embed_loss = preds[self.pred.index('embed_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label > 0)

        self.sum_metric += np.sum(embed_loss)
        self.num_inst += num_inst

        profile_off(tic, 'EmbedMetric metric')

class BGModelMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        name = 'BGModelMetric'
        super(BGModelMetric, self).__init__(name)
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        tic = profile_on()

        BG_model_loss = preds[self.pred.index('BG_model_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label > 0)

        self.sum_metric += np.sum(BG_model_loss)
        self.num_inst += num_inst

        profile_off(tic, 'EmbedMetric metric')

class RepsCLSMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        name = 'RepsCLSLoss'
        super(RepsCLSMetric, self).__init__(name)
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        tic = profile_on()

        reps_cls_loss = preds[self.pred.index('reps_cls_loss')].asnumpy()

        num_inst = self.cfg.network.REPS_PER_CLASS * self.cfg.dataset.NUM_CLASSES

        self.sum_metric += np.sum(reps_cls_loss)
        self.num_inst += num_inst

        profile_off(tic, 'RepsCLSMetric metric')

class ValRegMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        name = 'ValRegLoss'
        super(ValRegMetric, self).__init__(name)
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        tic = profile_on()

        val_reg_loss = preds[self.pred.index('val_reg_loss')].asnumpy()

        num_inst = self.cfg.network.NUM_ROIS_FOR_VAL_TRAIN

        self.sum_metric += np.sum(val_reg_loss)
        self.num_inst += num_inst

        profile_off(tic, 'RepsCLSMetric metric')

class ScoreHistMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        name = 'ScoreHistLoss'
        super(ScoreHistMetric, self).__init__(name)
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)
        self.cfg = cfg

    def update(self, labels, preds):
        tic = profile_on()

        score_hist_loss = preds[self.pred.index('score_hist_loss')].asnumpy()
        reg_hist = preds[self.pred.index('reg_hist')].asnumpy()

        num_inst = self.cfg.network.NUM_ROIS_FOR_VAL_TRAIN

        self.sum_metric += np.sum(score_hist_loss)
        self.num_inst += num_inst

        profile_off(tic, 'RepsCLSMetric metric')

