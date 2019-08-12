import numpy as np
import os
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.miscellaneous import get_GT_IoUs
import copy

def voc_ap(rec, prec, score, sc_part, use_07_metric=False):
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
        mscore = np.concatenate(([0.], score, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        pr_graph = {'mrec': mrec[i + 1], 'mpre': mpre[i + 1], 'mscore': mscore[i + 1]}
    return ap, pr_graph


class PerfStats():
    def __init__(self, Nclasses=1, Nslots=10000, is_valid=True, vet_score_thresh=0.55):
        self.valid = is_valid
        self.maxFpw = 0  # max FPW on some image in the query set
        self.nGT = 0  # number of GT objects so far
        self.d = 0  # current entry
        self.sc = np.zeros(Nslots)  # scores
        self.tp = np.zeros(Nslots)  # True Positives
        self.fp = np.zeros(Nslots)  # False Positives
        self.fpw = np.zeros(Nslots)  # False Positives - wrong class
        self.fpb = np.zeros(Nslots)  # False Positives - background box
        #self.require_difficult = difficult  # require that objects marked 'difficult' will be detected. Accept their detections anyway.
        self.vet_score_thresh = vet_score_thresh
        self.CM_thresh = np.linspace(0.1, 0.9, 9).tolist()
        self.CM = np.zeros([Nclasses, Nclasses,len(self.CM_thresh)],dtype=int)

    def assert_space(self):
        btch = 100000
        if self.sc.shape[0] < self.d + btch:
            self.sc = np.concatenate((self.sc, np.zeros(btch, )))
            self.tp = np.concatenate((self.tp, np.zeros(btch, )))
            self.fp = np.concatenate((self.fp, np.zeros(btch, )))
            self.fpw = np.concatenate((self.fpw, np.zeros(btch, )))
            self.fpb = np.concatenate((self.fpb, np.zeros(btch, )))



    def comp_epi_CM(self,scores_all,boxes_all,gt_boxes, gt_classes,epi_cats, ovthresh):
        all_cats = [0]+epi_cats
        Ndets = scores_all.shape[0]
        Nclasses = scores_all.shape[1]
        # for i_DetCat, DetCat in enumerate(epi_cats):
        #     cat_dets = np.hstack((boxes_all,scores_all[:,i_DetCat+1]))
        N_gt = gt_boxes.shape[0]
        gt_IoU_map = np.zeros((Ndets, N_gt))
        for i_det, box_det in enumerate(boxes_all):
            gt_IoU_map[i_det, :] = get_GT_IoUs(box_det, gt_boxes)
        true_class_indices = np.argmax(gt_IoU_map,axis=1)

        FG_idx = np.where(np.max(gt_IoU_map,axis=1) >= ovthresh)
        BG_idx = np.where(np.max(gt_IoU_map, axis=1) < ovthresh)

        true_class_indices_FG = true_class_indices[FG_idx]
        scores_FG = scores_all[FG_idx]
        scores_BG = scores_all[BG_idx]


        for i_thresh, thresh in enumerate(self.CM_thresh):
            FG_high_score_idx = np.where(np.max(scores_FG, axis=1) > thresh)
            BG_high_score_idx = np.where(np.max(scores_BG, axis=1) > thresh)
            true_class_indices_FG_high = true_class_indices_FG[FG_high_score_idx]
            scores_FG_high = scores_FG[FG_high_score_idx]
            scores_BG_high = scores_BG[BG_high_score_idx]
            hit_indices = np.argmax(scores_FG_high, axis=1)
            miss_indices = np.argmax(scores_BG_high, axis=1)
            # confusion matrix: for each detection roi: if its FG, the true class is its category; if its BG, the true class is 0 (background).
            # add 1 to row <true class>, column <argMax(scores)> for this detection

            for det_idx in miss_indices:
                self.CM[0, all_cats[det_idx],i_thresh ] += 1
                #print('object at cat 0 detected as cat {0}'.format(all_cats[det_idx]))

            for true_cls_idx, det_idx in zip(true_class_indices_FG_high,hit_indices):
                self.CM[gt_classes[true_cls_idx], all_cats[det_idx],i_thresh] += 1
                #print('object at cat {0} detected as cat {1}'.format(gt_classes[true_cls_idx],all_cats[det_idx]))


    def comp_epi_stats_m(self, q_dets, gt_boxes, gt_classes, epi_cats, ovthresh, err_score_thres=0,miss_count={},false_count={},true_count={}):
        # err_score_thres is the threshold for ploting the boxes as error case
        self.assert_space()
        N_gt = gt_boxes.shape[0]
        nTP = 0

        error_case = False #flag for ploting the boxes as error case

        for cat_dets, DetCat in zip(q_dets,epi_cats):  # for each class -------------------------------
        # for i_DetCat, DetCat in enumerate(epi_cats):  # for each class -------------------------------
        #     cat_dets = q_dets[i_DetCat]
            self.nGT += sum(gt_classes == DetCat)   # number of instances of this object

            Ndets = cat_dets.shape[0]
            if Ndets == 0:
                continue

            if not DetCat in miss_count:
                miss_count[DetCat]=0
                false_count[DetCat]=0
                true_count[DetCat]=0

            gt_IoU_map, score_map, hit_map = [np.zeros((Ndets, N_gt)) for _ in range(3)]
            for i_det, det in enumerate(cat_dets):
                gt_IoU_map[i_det, :] = get_GT_IoUs(det, gt_boxes)
                score_map[i_det, :] = det[-1]
            for col, gt_c in enumerate(gt_classes):
                hit_map[:, col] = gt_c == DetCat

            score_FG_map = np.multiply(score_map, gt_IoU_map >= ovthresh)
            score_BG_map = np.multiply(score_map, gt_IoU_map < ovthresh)
            hit_score_FG_map = np.multiply(score_FG_map, hit_map)
            best_hit_scores = np.max(hit_score_FG_map, axis=0)
            miss_score_FG_map = np.multiply(score_FG_map, 1 - hit_map)

            good_dets = np.max(hit_score_FG_map, axis=1)
            miss_score_FG_map[np.where(good_dets > 0)] = 0

            best_miss_scores = np.max(miss_score_FG_map, axis=0)
            score_BG_list = np.min(score_BG_map, axis=1)


            # TP
            for score in best_hit_scores: #dims = [1, Ngt]
                # for every GT box, take the best matching FG detection roi with the correct class, and set a True Positive with its score
                if score > 0:
                    self.set_score(score)
                    self.mark_TP()
                    self.d += 1
                    true_count[DetCat] += 1

            # FPW
            for score in best_miss_scores: #dims = [1, Ngt].
                # for every GT box, take the best matching FG detection roi (gt_IoU_map >= ovthresh) and set a False Positive with its score
                if score > 0:
                    self.set_score(score)
                    self.mark_FP()  # fp[d] = 1.
                    self.mark_FPW()  # fpw[d] = 1.
                    self.d += 1
                    false_count[DetCat] +=1


            # FPB
            for score in score_BG_list: # dims = [Ndets, 1]
                # for every detection roi, that was decided to be on background (gt_IoU_map < ovthresh), set a False Positive with its score
                if score > 0:
                    self.set_score(score)
                    self.mark_FP()  # fp[d] = 1.
                    self.mark_FPB()  # fpb[d] = 1.
                    self.d += 1
                    false_count[DetCat] +=1

            # flag for ploting the boxes as error case
            if np.any(best_miss_scores>err_score_thres):
                error_case = True # incorret class
            if np.any(score_BG_list>err_score_thres):
                error_case = True #high score outlier
            nTP += len(np.where(best_hit_scores>err_score_thres)[0])
            miss_count[DetCat] += np.sum(hit_map)-len(np.where(best_hit_scores>err_score_thres)[0])

        if nTP<N_gt:
            error_case = True # not all GT were detected
        # self.set_img_recAtK(d0, nGT_prev)

        return error_case, miss_count, true_count, false_count

    def set_score(self, sc):
        self.sc[self.d] = sc

    def get_score(self):
        return self.sc[0:self.d]

    def mark_TP(self):
        self.tp[self.d] = 1

    def get_TP(self):
        return self.tp[0:self.d]

    def mark_FP(self):
        self.fp[self.d] = 1

    def get_FP(self):
        return self.fp[0:self.d]

    def mark_FPW(self):
        self.fpw[self.d] = 1

    def get_FPW(self):
        return self.fpw[0:self.d]

    def mark_FPB(self):
        self.fpb[self.d] = 1

    def get_FPB(self):
        return self.fpb[0:self.d]


    def compute_stats(self, start_idx=0, use_nGT=-1):
        if start_idx >= self.d:
            return [None for _ in range(6)]
        sorted_inds = np.argsort(-self.sc[start_idx:self.d])  # ascending
        tp_part = self.tp[sorted_inds]
        fp_part = self.fp[sorted_inds]
        sc_part = self.sc[sorted_inds]
        fp_wrong_part = self.fpw[sorted_inds]
        fp_bkgnd_part = self.fpb[sorted_inds]
        tp_acc = np.cumsum(tp_part)
        fp_acc = np.cumsum(fp_part)
        fp_wrong_acc = np.cumsum(fp_wrong_part)
        fp_bkgnd_acc = np.cumsum(fp_bkgnd_part)
        if use_nGT >= 0:
            nGT = use_nGT
        else:
            nGT = self.nGT
        rec_part = tp_acc / float(self.nGT)
        rec_class_agn_part = (tp_acc+fp_wrong_acc)/ float(self.nGT)
        prec_part = tp_acc / np.maximum(tp_acc + fp_acc, np.finfo(np.float64).eps)  # avoid division by zero
        prec_w_part = tp_acc / np.maximum(rec_class_agn_part + fp_wrong_acc, np.finfo(np.float64).eps)  # avoid division by zero

        score_part = sc_part
        use_07_metric = False
        ap, pr_graph = voc_ap(rec_part, prec_part,sc_part, use_07_metric)
        ap_w, pr_w_graph = voc_ap(rec_class_agn_part, prec_w_part, sc_part, use_07_metric)

        tot_tp = tp_acc[-1]
        tot_fp = fp_acc[-1]
        tot_fp_wrong = fp_wrong_acc[-1]
        tot_fp_bkgnd = fp_bkgnd_acc[-1]
        if nGT > 0:
            recall = rec_part[-1]  # tot_tp / self.nGT # recall @ score thresh
        else:
            recall = 0
        pr_data = [[ap,pr_graph],[ap_w,pr_w_graph]]
        return tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data

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
        sorted_inds = np.argsort(-sc[start_idx:d])
        tp_part = tp[sorted_inds]
        fp_part = fp[sorted_inds]
        sc_part = sc[sorted_inds]
        fp_wrong_part = fpw[sorted_inds]
        fp_bkgnd_part = fpb[sorted_inds]
        tp_acc = np.cumsum(tp_part)
        fp_acc = np.cumsum(fp_part)
        fp_wrong_acc = np.cumsum(fp_wrong_part)
        fp_bkgnd_acc = np.cumsum(fp_bkgnd_part)
        rec_part = tp_acc / float(nGT)
        prec_part = tp_acc / np.maximum(tp_acc + fp_acc, np.finfo(np.float64).eps)  # avoid division by zero
        use_07_metric = False
        ap,pr_graph = voc_ap(rec_part, prec_part, sc_part,use_07_metric)
        tot_tp = tp_acc[-1]
        tot_fp = fp_acc[-1]
        tot_fp_wrong = fp_wrong_acc[-1]
        tot_fp_bkgnd = fp_bkgnd_acc[-1]
        if nGT > 0:
            recall = tot_tp / nGT
        else:
            recall = 0
        return tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap,pr_graph

    def isvalid(self):
        return self.valid

    def merge_stats_ext(self, stats):
        # stats[0] = [sc, tp, fp, fpw, fpb]
        d = stats[2]
        sc = stats[0][0, :]
        tp = stats[0][1, :]
        fp = stats[0][2, :]
        fpw = stats[0][3, :]
        fpb = stats[0][4, :]
        # left = nd - stats[0].shape[1]
        # if left>0:
        #     sc = np.concatenate((sc,np.zeros(left)))
        #     tp = np.concatenate((tp, np.zeros(left)))
        #     fp = np.concatenate((fp, np.zeros(left)))
        #     fpw = np.concatenate((fpw, np.zeros(left)))
        #     fpb = np.concatenate((fpb, np.zeros(left)))
        nGT = stats[1]


        if self.valid:  # concat the provided object
            self.sc = np.concatenate((self.get_score(), sc))
            self.tp = np.concatenate((self.get_TP(), tp))
            self.fp = np.concatenate((self.get_FP(), fp))
            self.fpw = np.concatenate((self.get_FPW(), fpw))
            self.fpb = np.concatenate((self.get_FPB(), fpb))
            self.nGT += nGT
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

    def print_stats(self,graph_img_fname, title_prefix=''):

        if self.d <= 0:
            print('No statistics were gathered.')
            return
        if self.nGT == 0 and use_nGT == -1:
            logger.info('#Dets: {0}, #GT: {1}'.format(0, self.nGT))
            return
        # statistics
        tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data = self.compute_stats()
        ap = pr_data[0][0]
        pr_graph =pr_data[0][1]
        ap_w = pr_data[0][0]
        pr_w_graph = pr_data[0][1]
        # RP curve -----------------------------------------------
        fig = plt.figure(1)
        plt.cla()
        plt.plot(pr_graph['mrec'],pr_graph['mpre'],linewidth=2.0)
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        for i in range(0,len(pr_graph['mscore']),4):
            plt.text(pr_graph['mrec'][i], pr_graph['mpre'][i], '{0:.2f}'.format(pr_graph['mscore'][i]))
        plt.title(title_prefix+'TP={0} FP_conf={1} FP_bkgnd={2} AP={3:.3f}'.format(tot_tp,tot_fp_wrong,tot_fp_bkgnd,ap))
        plt.grid(True)
        fig.savefig(graph_img_fname+'PR_graph.jpg')

        # confusion matrix -----------------------------------------------
        #
        fig = plt.figure(2)
        plt.cla()
        plt.imshow(self.CM[:,:,3])
        f_path, f_name = os.path.split(graph_img_fname)
        fig.savefig(graph_img_fname+'_confusion_matrix.jpg')

        if False: #class agnostic
            fig = plt.figure(2)
            plt.cla()
            plt.plot(pr_graph['mrec'],pr_graph['mpre'],linewidth=2.0)
            plt.xlabel('Recall - class agn.')
            plt.ylabel('Precision - class agn.')
            for i in range(0,len(pr_w_graph['mscore']),4):
                plt.text(pr_w_graph['mrec'][i], pr_w_graph['mpre'][i], '{0:.2f}'.format(pr_w_graph['mscore'][i]))
            plt.title(title_prefix+' class agn. AP={0:.3f}'.format(ap_w))
            plt.grid(True)
            f_path, f_name = os.path.split(graph_img_fname)
            fig.savefig(os.path.join(f_path,'class_agn_'+f_name))

    def print_perf(self, logger, prefix, start_idx=0, use_nGT=-1):
        # logger.info('== performance stats: ============================================================================================')
        if self.d <= 0:
            logger.info('No statistics were gathered.')
            return
        if self.nGT == 0 and use_nGT == -1:
            logger.info('#Dets: {0}, #GT: {1}'.format(0, self.nGT))
            return
        tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data = self.compute_stats(start_idx=start_idx, use_nGT=use_nGT)
        ap = pr_data[0][0]
        pr_graph =pr_data[0][1]
        ap_w = pr_data[0][0]
        pr_w_graph = pr_data[0][1]
        logger.info(prefix + ' #Dets: {0}, #GT: {1} TP: {2} FP: {3} = {4} wrong + {5} bkgnd  Recall: {6:.3f} AP: {7:.3f}' \
                    .format(self.d, self.nGT, np.int(tot_tp), np.int(tot_fp), np.int(tot_fp_wrong), np.int(tot_fp_bkgnd), recall, ap))

        temp = np.flip(np.abs(pr_graph['mpre']-0.80))
        idx=len(temp)-np.argmin(temp)-1
        score_th = pr_graph['mscore'][idx]

        return recall,ap,score_th  # chao add

    def print_perf_ext(self, logger, prefix, stats, start_idx=0):
        logger.info('== performance stats: ============================================================================================')
        nGT = stats[5]
        d = stats[6]
        if d <= 0:
            logger.info('No statistics were gathered.')
            return
        if nGT == 0:
            logger.info('#Dets: {0}, #GT: {1}'.format(0, nGT))
            return
        tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data = self.compute_stats_ext(stats, start_idx=start_idx)
        ap = pr_data[0][0]
        pr_graph =pr_data[0][1]
        ap_w = pr_data[0][0]
        pr_w_graph = pr_data[0][1]
        logger.info(prefix + ' #Dets: {0}, #GT: {1} TP: {2} FP: {3} = {4} wrong + {5} bkgnd  Recall: {6:.3f} AP: {7:.3f}'.format(d, nGT, \
                     np.int(tot_tp), np.int(tot_fp),np.int(tot_fp_wrong), np.int(tot_fp_bkgnd), recall,ap))
        return

    def save_stats(self, store_stats_fname):

        np.savez(store_stats_fname, stats=[self.sc, self.tp, self.fp, self.fpw, self.fpb, self.nGT, self.d, self.CM])

    def get_stats(self):
        stat_array = np.expand_dims(self.sc[:self.d], axis=0)
        stat_array = np.concatenate((stat_array, np.expand_dims(self.tp[:self.d], axis=0)), axis=0)
        stat_array = np.concatenate((stat_array, np.expand_dims(self.fp[:self.d], axis=0)), axis=0)
        stat_array = np.concatenate((stat_array, np.expand_dims(self.fpw[:self.d], axis=0)), axis=0)
        stat_array = np.concatenate((stat_array, np.expand_dims(self.fpb[:self.d], axis=0)), axis=0)
        stat_array = stat_array.astype(np.float16)
        return [stat_array, self.nGT, self.d, self.CM]  # [, , self.fp[:self.d], self.fpw[:self.d], self.fpb[:self.d], self.nGT, self.d, self.img_recAtK]
