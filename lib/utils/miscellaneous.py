import numpy as np
import os
import cv2
def vis_boxes(im, bboxes, classes, scale=1.0, save_file_path='temp.png'):
    import matplotlib.pyplot as plt
    from random import random as rand
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)

    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for cls_idx, cls_name in enumerate(classes):

        for bbox in bboxes:
            bbox = bbox[1:] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

    plt.show()

    fig.savefig(save_file_path)

    return im



def get_cat_names_ords(class_list_fname):
    with open(class_list_fname, 'r') as fid:
        cat_names = [x.strip() for x in fid.readlines()]
        cat_ords = range(1, len(cat_names) + 1)
        Ncats = len(cat_names)
        ord2name = {};
        name2ord = {}
        for ord, name in zip(cat_ords, cat_names):
            name2ord[name] = ord
            ord2name[ord] = name
    return cat_names, cat_ords, ord2name,name2ord


class BWlists():
    def __init__(self,bwlists_fname, multi=True):
        self.bwlists_fname = bwlists_fname
        if not os.path.exists(self.bwlists_fname):
            black_list = {}
            white_list = {}
            np.savez(self.bwlists_fname, black_list=black_list, white_list=white_list)
        self.multi=multi
        if self.multi: # no reading/writing to the file during the process
            bwlist_fnames = np.load(self.bwlists_fname)
            self.black_list = bwlist_fnames['black_list'].item()
            self.white_list = bwlist_fnames['white_list'].item()
            self.new_black_indices = []
            self.new_white_indices = []
    def check_idx(self,idx,list_type):
        if self.multi:
            if list_type=='black':
                return idx in self.black_list.keys()
            if list_type == 'white':
                return idx in self.white_list.keys()
        else:
            bwlist_fnames = np.load(self.bwlists_fname)
            black_list = bwlist_fnames['black_list'].item()
            white_list = bwlist_fnames['white_list'].item()
            if list_type=='black':
                return idx in black_list.keys()
            if list_type == 'white':
                return idx in white_list.keys()

    def set_idx(self,idx,list_type):
        if self.multi:
            if list_type=='black':
                self.black_list[idx] = 1
                self.new_black_indices += [idx]
            if list_type == 'white':
                self.white_list[idx] = 1
                self.new_white_indices += [idx]
        else:
            bwlist_fnames = np.load(self.bwlists_fname)
            black_list = bwlist_fnames['black_list'].item()
            white_list = bwlist_fnames['white_list'].item()
            if list_type=='black':
                black_list[idx] = 1
            if list_type == 'white':
                white_list[idx] = 1
            np.savez(self.bwlists_fname, black_list=black_list, white_list=white_list)

    def save_lists(self):
        bwlist_fnames = np.load(self.bwlists_fname)
        black_list = bwlist_fnames['black_list'].item()
        white_list = bwlist_fnames['white_list'].item()
        for idx in self.new_black_indices:
            black_list[idx] = 1
        for idx in self.new_white_indices:
            white_list[idx] = 1
        np.savez(self.bwlists_fname, black_list=self.black_list, white_list=self.white_list)

def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def get_GT_IoUs(det, bbgt):
    bb = det[:4]
    ixmin = np.maximum(bbgt[:, 0], bb[0])
    iymin = np.maximum(bbgt[:, 1], bb[1])
    ixmax = np.minimum(bbgt[:, 2], bb[2])
    iymax = np.minimum(bbgt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (bbgt[:, 2] - bbgt[:, 0] + 1.) *
           (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

def get_GT_IoUs_batch(dets, bbgt):
    overlaps = np.zeros((len(dets),len(bbgt)))
    for i_det,det in enumerate(dets):
        overlaps[i_det,:] = get_GT_IoUs(det, bbgt)
    return overlaps


def get_nearest_GTbox(det, bbgt):
    bb = det[:4]
    # compute overlaps of all GT boxes in this image with this one detection bbox
    # intersection
    ixmin = np.maximum(bbgt[:, 0], bb[0])
    iymin = np.maximum(bbgt[:, 1], bb[1])
    ixmax = np.minimum(bbgt[:, 2], bb[2])
    iymax = np.minimum(bbgt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (bbgt[:, 2] - bbgt[:, 0] + 1.) *
           (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)  # IoU value of maximal overlap
    jmax = np.argmax(overlaps)  # index of maximal overlap in GT lost
    return ovmax, jmax

def cos_sim_2_dist_generic(cos_sim, x=None, y=None, x_is_norm=True, y_is_norm=True):
    # returns distance squared
    # cos_sim = [#vectors x, #vectors y]
    # x = [cfg.network.EMBEDDING_DIM, #vectors x]
    # y = [cfg.network.EMBEDDING_DIM, #vectors y]

    if x_is_norm:
        x_norm = 1
    else:
        assert (x is not None), 'if x is not L2 normalized then x must be provided'
        x_norm = np.sum(np.square(x), axis=0, keepdims=True).T

    if y_is_norm:
        y_norm = 1
    else:
        assert (y is not None), 'if y is not L2 normalized then y must be provided'
        y_norm = np.sum(np.square(y), axis=0, keepdims=True).T

    dist =y_norm+x_norm-2*cos_sim

    return dist

class PerfStats():
    def __init__(self, Nslots, difficult=True, vet_score_thresh=0.55):
        if Nslots == 0:
            self.valid = False
            Nslots = 1  # just for placeholders
        else:
            self.valid = True

        self.sc = np.zeros(Nslots)  # scores
        self.tp = np.zeros(Nslots)  # True Positives
        self.fp = np.zeros(Nslots)  # False Positives
        self.fpw = np.zeros(Nslots)  # False Positives - wrong class
        self.fpb = np.zeros(Nslots)  # False Positives - background box
        self.maxFpw = 0  # max FPW on some image in the query set
        self.nGT = 0  # number of GT objects so far
        self.d = -1  # current entry
        self.require_difficult = difficult  # require that objects marked 'difficult' will be detected. Accept their detections anyway.
        self.vet_score_thresh = vet_score_thresh
        self.img_recAtK = []



    def comp_epi_stats_m(self,d,q_dets,gt_boxes,gt_classes,epi_cats,ovthresh):
        d0 = d
        nGT_prev = self.nGT
        N_gt = gt_boxes.shape[0]
        for i_DetCat, DetCat in enumerate(epi_cats):  # for each class -------------------------------
            cat_dets = q_dets[i_DetCat]
            n_objects = sum(gt_classes == DetCat)
            self.update_nGT(n_objects)  # number of instances of this object
            Ndets = cat_dets.shape[0]
            if Ndets==0:
                continue
            gt_IoU_map,score_map,hit_map = [np.zeros((Ndets, N_gt)) for _ in range(3)]
            for i_det, det in enumerate(cat_dets):
                gt_IoU_map[i_det,:] = get_GT_IoUs(det,gt_boxes)
                score_map[i_det,:] = det[-1]
            for col, gt_c in enumerate(gt_classes):
                hit_map[:,col] = gt_c==DetCat

            score_FG_map = np.multiply(score_map, gt_IoU_map >= ovthresh)
            score_BG_map = np.multiply(score_map, gt_IoU_map < ovthresh)
            hit_score_FG_map =  np.multiply(score_FG_map,hit_map)
            best_hit_scores = np.max(hit_score_FG_map,axis=0)
            miss_score_FG_map = np.multiply(score_FG_map,1-hit_map)

            good_dets = np.max(hit_score_FG_map, axis=1)
            miss_score_FG_map[np.where(good_dets > 0)] = 0

            best_miss_scores = np.max(miss_score_FG_map, axis=0)
            score_BG_list = np.min(score_BG_map,axis=1)
            #GT

            # if n_objects:
            #     print('adding {0} objects from cat {1}. total {2} objects'.format(n_objects, DetCat, self.nGT))
            # TP
            for score in best_hit_scores:
                if score>0:
                    self.set_score(score, d)
                    self.mark_TP(d)
                    d+=1

            #FPW
            for score in best_miss_scores:
                if score>0:
                    self.set_score(score, d)
                    self.mark_FP(d)  # fp[d] = 1.
                    self.mark_FPW(d)  # fpw[d] = 1.
                    d += 1

            #FPB
            for score in score_BG_list:
                if score>0:
                    self.set_score(score, d)
                    self.mark_FP(d)  # fp[d] = 1.
                    self.mark_FPB(d)  # fpb[d] = 1.
                    d += 1
        #self.set_img_recAtK(d0, nGT_prev)
        return d

    def comp_epi_stats_s(self,d, q_dets, gt_boxes, gt_classes, epi_cats, ovthresh):
        d0 = d
        relevant_instances = [i for i in gt_classes if i in epi_cats]
        self.update_nGT(len(relevant_instances))
        for iCls, cls_idx in enumerate(epi_cats):  # for each class -------------------------------
            cls_dets = q_dets[iCls]
            if cls_dets.size > 0:
                for det in cls_dets:  # for each bbox of this class
                    self.set_score(det[-1], d)  # sc[d] = det[-1]
                    ovmax, jmax = get_nearest_GTbox(det,gt_boxes)
                    if ovmax > ovthresh:
                        if gt_classes[jmax] == cls_idx:
                            self.mark_TP(d)  # tp[d] = 1.  # True Positive.
                            if True:
                                gt_boxes[jmax] = gt_boxes[jmax] + 10000
                        else:  # wrong class
                            self.mark_FP(d)  # fp[d] = 1.
                            self.mark_FPW(d)  # fpw[d] = 1.
                    else:  # no GT for this detection.
                        self.mark_FP(d)  # fp[d] = 1.
                        self.mark_FPB(d)  # fpb[d] = 1.
                    d += 1
        #self.set_img_recAtK(d0,nGT_prev)
        return d



    def update_nGT(self, num_gt_instances):
        self.nGT += num_gt_instances

    def set_score(self, sc, d):
        btch = 10000
        if self.sc.shape[0]<d+10:
            self.sc = np.concatenate((self.sc,np.zeros(btch,)))
            self.tp = np.concatenate((self.tp, np.zeros(btch,)))
            self.fp = np.concatenate((self.fp, np.zeros(btch,)))
            self.fpw = np.concatenate((self.fpw, np.zeros(btch,)))
            self.fpb = np.concatenate((self.fpb, np.zeros(btch,)))

        self.sc[d] = sc
        self.d = d

    def get_score(self):
        return self.sc[0:self.d + 1]

    def mark_TP(self, d):
        self.tp[d] = 1
        self.d = d

    def get_TP(self):
        return self.tp[0:self.d + 1]

    def mark_FP(self, d):
        self.fp[d] = 1
        self.d = d

    def get_FP(self):
        return self.fp[0:self.d + 1]

    def mark_FPW(self, d):
        self.fpw[d] = 1
        self.d = d

    def get_FPW(self):
        return self.fpw[0:self.d + 1]

    def mark_FPB(self, d):
        self.fpb[d] = 1
        self.d = d

    def get_FPB(self):
        return self.fpb[0:self.d + 1]

    def update_maxFpw(self, d0, d):
        sc_img = self.sc[d0:d]
        fpb_img = self.fpb[d0:d]
        fpb_img_bad = [fpbi for sci, fpbi in zip(sc_img, fpb_img) if sci > self.vet_score_thresh]
        self.maxFpw = max(self.maxFpw, sum(fpb_img_bad))

    def compute_stats(self, start_idx, use_nGT=-1):
        if start_idx > self.d:
            return [None for _ in range(6)]
        sorted_inds = np.argsort(-self.sc[start_idx:self.d + 1]) # ascending
        tp_part = self.tp[sorted_inds]
        fp_part = self.fp[sorted_inds]
        fp_wrong_part = self.fpw[sorted_inds]
        fp_bkgnd_part = self.fpb[sorted_inds]
        tp_acc = np.cumsum(tp_part)
        fp_acc = np.cumsum(fp_part)
        fp_wrong_acc = np.cumsum(fp_wrong_part)
        fp_bkgnd_acc = np.cumsum(fp_bkgnd_part)
        if use_nGT>=0:
            nGT = use_nGT
        else:
            nGT = self.nGT
        rec_part = tp_acc / float(self.nGT)
        prec_part = tp_acc / np.maximum(tp_acc + fp_acc, np.finfo(np.float64).eps)  # avoid division by zero
        use_07_metric = False
        ap = voc_ap(rec_part, prec_part, use_07_metric)
        tot_tp = tp_acc[-1]
        tot_fp = fp_acc[-1]
        tot_fp_wrong = fp_wrong_acc[-1]
        tot_fp_bkgnd = fp_bkgnd_acc[-1]
        if nGT > 0:
            recall = rec_part[-1] #tot_tp / self.nGT # recall @ score thresh
        else:
            recall = 0

        # compute recall@K
        return tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap

    def set_img_recAtK(self, d0,nGT_prev):
        nGT_img = self.nGT-nGT_prev
        sorted_inds = np.argsort(-self.sc[d0:self.d])
        tp_img = self.tp[d0:self.d]
        tp_part = tp_img[sorted_inds]
        tp_acc = np.cumsum(tp_part)
        rec_part = tp_acc / float(nGT_img)
        K_set = [5,10,20,50,100]
        recAtK_set = [ -1 for _ in K_set] # initialize
        for iK, K in enumerate(K_set):
            if len(sorted_inds)>=K:
                recAtK_set[iK] = rec_part[K-1] # top K true positives
        self.img_recAtK+=[recAtK_set]

    def get_mean_recAtK(self):
        img_recAtK = np.array(self.img_recAtK)
        mean_img_recAtK = -1*np.ones(img_recAtK.shape[1])
        for col in range(img_recAtK.shape[1]):
            subset = np.where(img_recAtK[:,col]>0)
            if len(subset[0])>0:
                mean_img_recAtK[col] = np.mean(img_recAtK[subset,col])
        return mean_img_recAtK

    def compute_stats_ext(self, stats, start_idx=0):
        sc = stats[0]
        tp = stats[1]
        fp = stats[2]
        fpw = stats[3]
        fpb = stats[4]
        nGT = stats[5]
        d = stats[6]
        if start_idx > d:
            return [[] for _ in range(6)]
        sorted_inds = np.argsort(-sc[start_idx:d + 1])
        tp_part = tp[sorted_inds]
        fp_part = fp[sorted_inds]
        fp_wrong_part = fpw[sorted_inds]
        fp_bkgnd_part = fpb[sorted_inds]
        tp_acc = np.cumsum(tp_part)
        fp_acc = np.cumsum(fp_part)
        fp_wrong_acc = np.cumsum(fp_wrong_part)
        fp_bkgnd_acc = np.cumsum(fp_bkgnd_part)
        rec_part = tp_acc / float(nGT)
        prec_part = tp_acc / np.maximum(tp_acc + fp_acc, np.finfo(np.float64).eps)  # avoid division by zero
        use_07_metric = False
        ap,pr_graph = voc_ap(rec_part, prec_part, use_07_metric)
        tot_tp = tp_acc[-1]
        tot_fp = fp_acc[-1]
        tot_fp_wrong = fp_wrong_acc[-1]
        tot_fp_bkgnd = fp_bkgnd_acc[-1]
        if nGT > 0:
            recall = tot_tp / nGT
        else:
            recall = 0
        return tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap

    def isvalid(self):
        return self.valid

    def merge_stats_ext(self, stats):
        # stats[0] = [sc, tp, fp, fpw, fpb]
        d = stats[2]
        sc =stats[0][0,:]
        tp = stats[0][1,:]
        fp = stats[0][2,:]
        fpw = stats[0][3,:]
        fpb = stats[0][4,:]
        # left = nd - stats[0].shape[1]
        # if left>0:
        #     sc = np.concatenate((sc,np.zeros(left)))
        #     tp = np.concatenate((tp, np.zeros(left)))
        #     fp = np.concatenate((fp, np.zeros(left)))
        #     fpw = np.concatenate((fpw, np.zeros(left)))
        #     fpb = np.concatenate((fpb, np.zeros(left)))
        nGT = stats[1]
        img_recAtK = stats[3]

        if self.valid:  # concat the provided object
            self.sc = np.concatenate((self.get_score(), sc))
            self.tp = np.concatenate((self.get_TP(), tp))
            self.fp = np.concatenate((self.get_FP(), fp))
            self.fpw = np.concatenate((self.get_FPW(), fpw))
            self.fpb = np.concatenate((self.get_FPB(), fpb))
            self.nGT += nGT
            self.img_recAtK.extend(img_recAtK)
            self.d += d + 1
        else:  # copy the provided object
            self.sc = sc
            self.tp = tp
            self.fp = fp
            self.fpw = fpw
            self.fpb = fpb
            self.nGT = nGT
            self.maxFpw = 0
            self.d = d
            self.valid = True
            self.img_recAtK = img_recAtK

    def merge_stats(self, perf_stats):
        if not perf_stats.isvalid:
            return
        if self.valid:  # concat the provided object
            self.sc = np.concatenate((self.get_score(), perf_stats.get_score()))
            self.tp = np.concatenate((self.get_TP(), perf_stats.get_TP()))
            self.fp = np.concatenate((self.get_FP(), perf_stats.get_FP()))
            self.fpw = np.concatenate((self.get_FPW(), perf_stats.get_FPW()))
            self.fpb = np.concatenate((self.get_FPB(), perf_stats.get_FPB()))
            self.nGT += perf_stats.nGT
            self.maxFpw = max(self.maxFpw, perf_stats.maxFpw)
            self.d += perf_stats.d + 1
        else:  # copy the provided object
            self.sc = perf_stats.get_score()
            self.tp = perf_stats.get_TP()
            self.fp = perf_stats.get_FP()
            self.fpw = perf_stats.get_FPW()
            self.fpb = perf_stats.get_FPB()
            self.nGT = perf_stats.nGT
            self.maxFpw = perf_stats.maxFpw
            self.d = perf_stats.d
            self.valid = True

    def print_perf(self, logger, prefix, start_idx=0,use_nGT=-1):
        #logger.info('== performance stats: ============================================================================================')
        if self.d < 0:
            logger.info('No statistics were gathered.')
            return
        if self.nGT == 0 and use_nGT==-1:
            logger.info('#Dets: {0}, #GT: {1}'.format(0, self.nGT))
            return
        tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap = self.compute_stats(start_idx=start_idx,use_nGT=use_nGT)
        #recAtK_set = self.get_mean_recAtK()
        # logger.info(prefix + ' #Dets: {0}, #GT: {1} TP: {2} FP: {3} = {4} wrong + {5} bkgnd  Recall: {6:.3f} Rec@K: [5:{7:.3f} 10:{8:.3f} 20:{9:.3f} 50:{10:.3f}] AP: {11:.3f}'\
        #             .format(self.d+1, self.nGT, np.int(tot_tp), np.int(tot_fp), np.int(tot_fp_wrong), np.int(tot_fp_bkgnd), recall, recAtK_set[0], recAtK_set[1],recAtK_set[2],recAtK_set[3], ap))
        logger.info(prefix + ' #Dets: {0}, #GT: {1} TP: {2} FP: {3} = {4} wrong + {5} bkgnd  Recall: {6:.3f} AP: {7:.3f}'\
                     .format(self.d+1, self.nGT, np.int(tot_tp), np.int(tot_fp), np.int(tot_fp_wrong), np.int(tot_fp_bkgnd), recall, ap))
        #logger.info(prefix + ' #Dets: {0}, #GT: {1} Recall: {2:.3f} AP: {3:.3f}'.format(self.d+1, self.nGT, recall, ap))

    def print_perf_ext(self, logger, prefix, stats, start_idx=0):
        logger.info('== performance stats: ============================================================================================')
        nGT = stats[5]
        d = stats[6]
        if d < 0:
            logger.info('No statistics were gathered.')
            return
        if nGT == 0:
            logger.info('#Dets: {0}, #GT: {1}'.format(0, nGT))
            return
        tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap = self.compute_stats_ext(stats,start_idx=start_idx)
        logger.info(prefix + ' #Dets: {0}, #GT: {1} TP: {2} FP: {3} = {4} wrong + {5} bkgnd  Recall: {6:.3f} AP: {7:.3f}'.format(d, nGT, \
            np.int(tot_tp), np.int(tot_fp), np.int(tot_fp_wrong),np.int(tot_fp_bkgnd), recall, ap))
        return

    def save_stats(self, store_stats_fname):

        np.savez(store_stats_fname, stats=[self.sc, self.tp, self.fp, self.fpw, self.fpb, self.nGT, self.d])

    def get_stats(self):
        stat_array = np.expand_dims(self.sc[:self.d+1], axis=0)
        stat_array = np.concatenate((stat_array,  np.expand_dims(self.tp[:self.d+1], axis=0)), axis=0)
        stat_array = np.concatenate((stat_array, np.expand_dims(self.fp[:self.d+1], axis=0)), axis=0)
        stat_array = np.concatenate((stat_array, np.expand_dims(self.fpw[:self.d+1], axis=0)), axis=0)
        stat_array = np.concatenate((stat_array, np.expand_dims(self.fpb[:self.d+1], axis=0)), axis=0)
        stat_array = stat_array.astype(np.float16)
        return [stat_array,self.nGT, self.d, self.img_recAtK] #[, , self.fp[:self.d], self.fpw[:self.d], self.fpb[:self.d], self.nGT, self.d, self.img_recAtK]

def get_disp_data(im_name, scores, boxes, nms,  score_thresh, CLASS_AGNOSTIC = True):
    import cv2
    dets_nms = []
    for j in range(1, scores.shape[1]):
        cls_scores = scores[:, j, np.newaxis]
        cls_boxes = boxes[:, 4:8] if CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        keep = nms(cls_dets)
        cls_dets = cls_dets[keep, :]
        cls_dets = cls_dets[cls_dets[:, -1] > score_thresh, :]
        dets_nms.append(cls_dets)

    # load image
    im = cv2.imread(im_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im, dets_nms

def flatten(list_lev2):
    if list_lev2 !=[] and type(list_lev2[0]) is list:
        list_lev1 = [i for lev1 in list_lev2 for i in lev1]
        return list_lev1
    else:
        return list_lev2

def configure_logging(log_filename):
    import logging
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    # Format for our loglines
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup file logging
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def assert_folder(folder):
    import os
    if not os.path.exists(folder):
        f_path, f_name = os.path.split(folder)
        if len(f_path)>0:
            assert_folder(f_path)
        os.mkdir(folder)
    return folder

def print_img(im,save_fname):
    import matplotlib.pyplot as plt
    from random import random as rand
    fig = plt.figure(1)
    fig.set_size_inches((2 * 8.5, 2 * 11), forward=False)
    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    fig.savefig(save_fname)

def compute_det_types(rois_feats,epi_cats,q_dets,gt_boxes,gt_classes,pars):
    det_types, IoU_values, score_values = [np.zeros((rois_feats.shape[0], len(epi_cats))) for _ in range(3)]
    for iCls, cls_idx in enumerate(epi_cats):  # for each class -------------------------------
        cls_dets = q_dets[iCls]
        if cls_dets.size > 0:
            for det in cls_dets:  # for each bbox of this class
                ovmax, jmax = get_nearest_GTbox(det, gt_boxes)
                IoU_values[int(det[4]), iCls] = ovmax
                score_values[int(det[4]), iCls] = det[-1]
                if ovmax > pars.ovthresh:
                    if gt_classes[jmax] == cls_idx:  # right class
                        det_types[int(det[4]), iCls] = 1
                    else:  # wrong class
                        det_types[int(det[4]), iCls] = 2
                else:  # background
                    det_types[int(det[4]), iCls] = 3
    return det_types,IoU_values,score_values

def strip_special_chars(st):
    for i in range(len(st)):
        if ord(st[i]) > 128:
            st = st.replace(st[i], 'a')
    st = st.replace('/','_')
    return st
