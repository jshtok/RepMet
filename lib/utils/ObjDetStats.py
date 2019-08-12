import numpy as np
import os
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils.miscellaneous import get_GT_IoUs
import copy
import cPickle
from FSD_common_lib import configure_logging,detName_2_imgName,imgName_2_detsName,imgName_2_dvisName,img_dets_CSV_2_A

# def voc_ap(rec, prec, score, sc_part, use_07_metric=False):
#     """
#     average precision calculations
#     [precision integrated to recall]
#     :param rec: recall
#     :param prec: precision
#     :param use_07_metric: 2007 metric is 11-recall-point based AP
#     :return: average precision
#     """
#     if use_07_metric:
#         ap = 0.
#         for t in np.arange(0., 1.1, 0.1):
#             if np.sum(rec >= t) == 0:
#                 p = 0
#             else:
#                 p = np.max(prec[rec >= t])
#             ap += p / 11.
#     else:
#         # append sentinel values at both ends
#         mrec = np.concatenate(([0.], rec, [1.]))
#         mpre = np.concatenate(([0.], prec, [0.]))
#         mscore = np.concatenate(([0.], score, [0.]))
#
#         # compute precision integration ladder
#         for i in range(mpre.size - 1, 0, -1):
#             mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
#
#         # look for recall value changes
#         i = np.where(mrec[1:] != mrec[:-1])[0]
#
#         # sum (\delta recall) * prec
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#         pr_graph = {'mrec': mrec[i + 1], 'mpre': mpre[i + 1], 'mscore': mscore[i + 1]}
#     return ap, pr_graph
# from utils.PerfStats import PerfStats

# class PerfStats_internal():
#     def __init__(self, Nclasses, Nslots=10000, is_valid=True, vet_score_thresh=0.55):
#         self.valid = is_valid
#         self.maxFpw = 0  # max FPW on some image in the query set
#         self.nGT = 0  # number of GT objects so far
#         self.d = 0  # current entry
#         self.sc = np.zeros(Nslots)  # scores
#         self.tp = np.zeros(Nslots)  # True Positives
#         self.fp = np.zeros(Nslots)  # False Positives
#         self.fpw = np.zeros(Nslots)  # False Positives - wrong class
#         self.fpb = np.zeros(Nslots)  # False Positives - background box
#         #self.require_difficult = difficult  # require that objects marked 'difficult' will be detected. Accept their detections anyway.
#         self.vet_score_thresh = vet_score_thresh
#         self.CM_thresh = np.linspace(0.1, 0.9, 9).tolist()
#         self.CM = np.zeros([Nclasses, Nclasses,len(self.CM_thresh)],dtype=int)
#
#     def assert_space(self):
#         btch = 10000
#         if self.sc.shape[0] < self.d + btch:
#             self.sc = np.concatenate((self.sc, np.zeros(btch, )))
#             self.tp = np.concatenate((self.tp, np.zeros(btch, )))
#             self.fp = np.concatenate((self.fp, np.zeros(btch, )))
#             self.fpw = np.concatenate((self.fpw, np.zeros(btch, )))
#             self.fpb = np.concatenate((self.fpb, np.zeros(btch, )))
#
#     def comp_epi_CM(self,scores_all,boxes_all,gt_boxes, gt_classes,epi_cats, ovthresh):
#         all_cats = [0]+epi_cats
#         Ndets = scores_all.shape[0]
#         Nclasses = scores_all.shape[1]
#         # for i_DetCat, DetCat in enumerate(epi_cats):
#         #     cat_dets = np.hstack((boxes_all,scores_all[:,i_DetCat+1]))
#         N_gt = gt_boxes.shape[0]
#         gt_IoU_map = np.zeros((Ndets, N_gt))
#         for i_det, box_det in enumerate(boxes_all):
#             gt_IoU_map[i_det, :] = get_GT_IoUs(box_det, gt_boxes)
#         true_class_indices = np.argmax(gt_IoU_map,axis=1)
#
#         FG_idx = np.where(np.max(gt_IoU_map,axis=1) >= ovthresh)
#         BG_idx = np.where(np.max(gt_IoU_map, axis=1) < ovthresh)
#
#         true_class_indices_FG = true_class_indices[FG_idx]
#         scores_FG = scores_all[FG_idx]
#         scores_BG = scores_all[BG_idx]
#
#
#         for i_thresh, thresh in enumerate(self.CM_thresh):
#             FG_high_score_idx = np.where(np.max(scores_FG, axis=1) > thresh)
#             BG_high_score_idx = np.where(np.max(scores_BG, axis=1) > thresh)
#             true_class_indices_FG_high = true_class_indices_FG[FG_high_score_idx]
#             scores_FG_high = scores_FG[FG_high_score_idx]
#             scores_BG_high = scores_BG[BG_high_score_idx]
#             hit_indices = np.argmax(scores_FG_high, axis=1)
#             miss_indices = np.argmax(scores_BG_high, axis=1)
#             # confusion matrix: for each detection roi: if its FG, the true class is its category; if its BG, the true class is 0 (background).
#             # add 1 to row <true class>, column <argMax(scores)> for this detection
#
#             for det_idx in miss_indices:
#                 self.CM[0, all_cats[det_idx],i_thresh ] += 1
#                 #print('object at cat 0 detected as cat {0}'.format(all_cats[det_idx]))
#
#             for true_cls_idx, det_idx in zip(true_class_indices_FG_high,hit_indices):
#                 self.CM[gt_classes[true_cls_idx], all_cats[det_idx],i_thresh] += 1
#                 #print('object at cat {0} detected as cat {1}'.format(gt_classes[true_cls_idx],all_cats[det_idx]))
#
#     def comp_epi_stats_m(self, q_dets, gt_boxes, gt_ords, model_cat_ords, ovthresh):
#         self.assert_space()
#         N_gt = gt_boxes.shape[0]
#
#         for cat_dets, det_ord in zip(q_dets, model_cat_ords):  # for each class -------------------------------
#             self.nGT += sum(gt_ords == det_ord)   # number of instances of this object
#
#             Ndets = cat_dets.shape[0]
#             if Ndets == 0:
#                 continue
#             gt_IoU_map, score_map, hit_map = [np.zeros((Ndets, N_gt)) for _ in range(3)]
#             for i_det, det in enumerate(cat_dets):
#                 gt_IoU_map[i_det, :] = get_GT_IoUs(det, gt_boxes)
#                 score_map[i_det, :] = det[-1]
#             for col, gt_c in enumerate(gt_ords):
#                 hit_map[:, col] = gt_c == det_ord
#
#             score_FG_map = np.multiply(score_map, gt_IoU_map >= ovthresh)
#             score_BG_map = np.multiply(score_map, gt_IoU_map < ovthresh)
#             hit_score_FG_map = np.multiply(score_FG_map, hit_map)
#             best_hit_scores = np.max(hit_score_FG_map, axis=0)
#             miss_score_FG_map = np.multiply(score_FG_map, 1 - hit_map)
#             best_miss_scores = np.max(miss_score_FG_map, axis=0)
#             score_BG_list = np.min(score_BG_map, axis=1)
#
#
#             # TP
#             for score in best_hit_scores: #dims = [1, Ngt]
#                 # for every GT box, take the best matching FG detection roi with the correct class, and set a True Positive with its score
#                 if score > 0:
#                     self.set_score(score)
#                     self.mark_TP()
#                     self.d += 1
#
#             # FPW
#             for score in best_miss_scores: #dims = [1, Ngt].
#                 # for every GT box, take the best matching FG detection roi (gt_IoU_map >= ovthresh) and set a False Positive with its score
#                 if score > 0:
#                     self.set_score(score)
#                     self.mark_FP()  # fp[d] = 1.
#                     self.mark_FPW()  # fpw[d] = 1.
#                     self.d += 1
#
#             # FPB
#             for score in score_BG_list:
#                 # for every detection roi, that was decided to be on background (gt_IoU_map < ovthresh), set a False Positive with its score
#                 if score > 0:
#                     self.set_score(score)
#                     self.mark_FP()  # fp[d] = 1.
#                     self.mark_FPB()  # fpb[d] = 1.
#                     self.d += 1
#         # self.set_img_recAtK(d0, nGT_prev)
#
#
#     def set_score(self, sc):
#         self.sc[self.d] = sc
#
#     def get_score(self):
#         return self.sc[0:self.d]
#
#     def mark_TP(self):
#         self.tp[self.d] = 1
#
#     def get_TP(self):
#         return self.tp[0:self.d]
#
#     def mark_FP(self):
#         self.fp[self.d] = 1
#
#     def get_FP(self):
#         return self.fp[0:self.d]
#
#     def mark_FPW(self):
#         self.fpw[self.d] = 1
#
#     def get_FPW(self):
#         return self.fpw[0:self.d]
#
#     def mark_FPB(self):
#         self.fpb[self.d] = 1
#
#     def get_FPB(self):
#         return self.fpb[0:self.d]
#
#
#     def compute_stats(self, start_idx=0, use_nGT=-1):
#         if start_idx >= self.d:
#             return [None for _ in range(6)]
#         sorted_inds = np.argsort(-self.sc[start_idx:self.d])  # ascending
#         tp_part = self.tp[sorted_inds]
#         fp_part = self.fp[sorted_inds]
#         sc_part = self.sc[sorted_inds]
#         fp_wrong_part = self.fpw[sorted_inds]
#         fp_bkgnd_part = self.fpb[sorted_inds]
#         tp_acc = np.cumsum(tp_part)
#         fp_acc = np.cumsum(fp_part)
#         fp_wrong_acc = np.cumsum(fp_wrong_part)
#         fp_bkgnd_acc = np.cumsum(fp_bkgnd_part)
#         if use_nGT >= 0:
#             nGT = use_nGT
#         else:
#             nGT = self.nGT
#         rec_part = tp_acc / float(self.nGT)
#         rec_class_agn_part = (tp_acc+fp_wrong_acc)/ float(self.nGT)
#         prec_part = tp_acc / np.maximum(tp_acc + fp_acc, np.finfo(np.float64).eps)  # avoid division by zero
#         prec_w_part = tp_acc / np.maximum(rec_class_agn_part + fp_wrong_acc, np.finfo(np.float64).eps)  # avoid division by zero
#
#         score_part = sc_part
#         use_07_metric = False
#         ap, pr_graph = voc_ap(rec_part, prec_part,sc_part, use_07_metric)
#         ap_w, pr_w_graph = voc_ap(rec_class_agn_part, prec_w_part, sc_part, use_07_metric)
#
#         tot_tp = tp_acc[-1]
#         tot_fp = fp_acc[-1]
#         tot_fp_wrong = fp_wrong_acc[-1]
#         tot_fp_bkgnd = fp_bkgnd_acc[-1]
#         if nGT > 0:
#             recall = rec_part[-1]  # tot_tp / self.nGT # recall @ score thresh
#         else:
#             recall = 0
#         pr_data = [[ap,pr_graph],[ap_w,pr_w_graph]]
#         return tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data
#
#     def compute_stats_ext(self, stats, start_idx=0):
#         sc = stats[0]
#         tp = stats[1]
#         fp = stats[2]
#         fpw = stats[3]
#         fpb = stats[4]
#         nGT = stats[5]
#         d = stats[6]
#         if start_idx > d:
#             return [[] for _ in range(6)]
#         sorted_inds = np.argsort(-sc[start_idx:d])
#         tp_part = tp[sorted_inds]
#         fp_part = fp[sorted_inds]
#         sc_part = sc[sorted_inds]
#         fp_wrong_part = fpw[sorted_inds]
#         fp_bkgnd_part = fpb[sorted_inds]
#         tp_acc = np.cumsum(tp_part)
#         fp_acc = np.cumsum(fp_part)
#         fp_wrong_acc = np.cumsum(fp_wrong_part)
#         fp_bkgnd_acc = np.cumsum(fp_bkgnd_part)
#         rec_part = tp_acc / float(nGT)
#         prec_part = tp_acc / np.maximum(tp_acc + fp_acc, np.finfo(np.float64).eps)  # avoid division by zero
#         use_07_metric = False
#         ap,pr_graph = voc_ap(rec_part, prec_part, sc_part,use_07_metric)
#         tot_tp = tp_acc[-1]
#         tot_fp = fp_acc[-1]
#         tot_fp_wrong = fp_wrong_acc[-1]
#         tot_fp_bkgnd = fp_bkgnd_acc[-1]
#         if nGT > 0:
#             recall = tot_tp / nGT
#         else:
#             recall = 0
#         return tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, ap,pr_graph
#
#     def isvalid(self):
#         return self.valid
#
#     def merge_stats_ext(self, stats):
#         # stats[0] = [sc, tp, fp, fpw, fpb]
#         d = stats[2]
#         sc = stats[0][0, :]
#         tp = stats[0][1, :]
#         fp = stats[0][2, :]
#         fpw = stats[0][3, :]
#         fpb = stats[0][4, :]
#         # left = nd - stats[0].shape[1]
#         # if left>0:
#         #     sc = np.concatenate((sc,np.zeros(left)))
#         #     tp = np.concatenate((tp, np.zeros(left)))
#         #     fp = np.concatenate((fp, np.zeros(left)))
#         #     fpw = np.concatenate((fpw, np.zeros(left)))
#         #     fpb = np.concatenate((fpb, np.zeros(left)))
#         nGT = stats[1]
#
#
#         if self.valid:  # concat the provided object
#             self.sc = np.concatenate((self.get_score(), sc))
#             self.tp = np.concatenate((self.get_TP(), tp))
#             self.fp = np.concatenate((self.get_FP(), fp))
#             self.fpw = np.concatenate((self.get_FPW(), fpw))
#             self.fpb = np.concatenate((self.get_FPB(), fpb))
#             self.nGT += nGT
#             self.d += d + 1
#         else:  # copy the provided object
#             self.sc = sc
#             self.tp = tp
#             self.fp = fp
#             self.fpw = fpw
#             self.fpb = fpb
#             self.nGT = nGT
#             self.maxFpw = 0
#             self.d = d
#             self.valid = True
#
#     def merge_stats(self, perf_stats):
#         if not perf_stats.isvalid:
#             return
#         if self.valid:  # concat the provided object
#             self.sc = np.concatenate((self.get_score(), perf_stats.get_score()))
#             self.tp = np.concatenate((self.get_TP(), perf_stats.get_TP()))
#             self.fp = np.concatenate((self.get_FP(), perf_stats.get_FP()))
#             self.fpw = np.concatenate((self.get_FPW(), perf_stats.get_FPW()))
#             self.fpb = np.concatenate((self.get_FPB(), perf_stats.get_FPB()))
#             self.nGT += perf_stats.nGT
#             self.maxFpw = max(self.maxFpw, perf_stats.maxFpw)
#             self.d += perf_stats.d + 1
#         else:  # copy the provided object
#             self.sc = perf_stats.get_score()
#             self.tp = perf_stats.get_TP()
#             self.fp = perf_stats.get_FP()
#             self.fpw = perf_stats.get_FPW()
#             self.fpb = perf_stats.get_FPB()
#             self.nGT = perf_stats.nGT
#             self.maxFpw = perf_stats.maxFpw
#             self.d = perf_stats.d
#             self.valid = True
#
#     def print_stats(self,graph_img_fname, title_prefix=''):
#
#         if self.d <= 0:
#             print('No statistics were gathered.')
#             return
#         if self.nGT == 0 and use_nGT == -1:
#             logger.info('#Dets: {0}, #GT: {1}'.format(0, self.nGT))
#             return
#         # statistics
#         tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data = self.compute_stats()
#         ap = pr_data[0][0]
#         pr_graph =pr_data[0][1]
#         ap_w = pr_data[0][0]
#         pr_w_graph = pr_data[0][1]
#         # RP curve -----------------------------------------------
#         fig = plt.figure(1)
#         plt.cla()
#         plt.plot(pr_graph['mrec'],pr_graph['mpre'],linewidth=2.0)
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#
#         for i in range(0,len(pr_graph['mscore']),4):
#             plt.text(pr_graph['mrec'][i], pr_graph['mpre'][i], '{0:.2f}'.format(pr_graph['mscore'][i]))
#         plt.title(title_prefix+'TP={0} FP_conf={1} FP_bkgnd={2} AP={3:.3f}'.format(tot_tp,tot_fp_wrong,tot_fp_bkgnd,ap))
#         plt.grid(True)
#         fig.savefig(graph_img_fname+'PR_graph.jpg')
#
#         # confusion matrix -----------------------------------------------
#         #
#         fig = plt.figure(2)
#         plt.cla()
#         plt.imshow(self.CM[:,:,3])
#         f_path, f_name = os.path.split(graph_img_fname)
#         fig.savefig(graph_img_fname+'_confusion_matrix.jpg')
#
#         if False: #class agnostic
#             fig = plt.figure(2)
#             plt.cla()
#             plt.plot(pr_graph['mrec'],pr_graph['mpre'],linewidth=2.0)
#             plt.xlabel('Recall - class agn.')
#             plt.ylabel('Precision - class agn.')
#             for i in range(0,len(pr_w_graph['mscore']),4):
#                 plt.text(pr_w_graph['mrec'][i], pr_w_graph['mpre'][i], '{0:.2f}'.format(pr_w_graph['mscore'][i]))
#             plt.title(title_prefix+' class agn. AP={0:.3f}'.format(ap_w))
#             plt.grid(True)
#             f_path, f_name = os.path.split(graph_img_fname)
#             fig.savefig(os.path.join(f_path,'class_agn_'+f_name))
#
#     def print_perf(self, logger, prefix, start_idx=0, use_nGT=-1):
#         # logger.info('== performance stats: ============================================================================================')
#         if self.d <= 0:
#             logger.info('No statistics were gathered.')
#             return
#         if self.nGT == 0 and use_nGT == -1:
#             logger.info('#Dets: {0}, #GT: {1}'.format(0, self.nGT))
#             return
#         tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data = self.compute_stats(start_idx=start_idx, use_nGT=use_nGT)
#         ap = pr_data[0][0]
#         pr_graph =pr_data[0][1]
#         ap_w = pr_data[0][0]
#         pr_w_graph = pr_data[0][1]
#         logger.info(prefix + ' #Dets: {0}, #GT: {1} TP: {2} FP: {3} = {4} wrong + {5} bkgnd  Recall: {6:.3f} AP: {7:.3f}' \
#                     .format(self.d, self.nGT, np.int(tot_tp), np.int(tot_fp), np.int(tot_fp_wrong), np.int(tot_fp_bkgnd), recall, ap))
#
#     def print_perf_ext(self, logger, prefix, stats, start_idx=0):
#         logger.info('== performance stats: ============================================================================================')
#         nGT = stats[5]
#         d = stats[6]
#         if d <= 0:
#             logger.info('No statistics were gathered.')
#             return
#         if nGT == 0:
#             logger.info('#Dets: {0}, #GT: {1}'.format(0, nGT))
#             return
#         tot_tp, tot_fp, tot_fp_wrong, tot_fp_bkgnd, recall, pr_data = self.compute_stats_ext(stats, start_idx=start_idx)
#         ap = pr_data[0][0]
#         pr_graph =pr_data[0][1]
#         ap_w = pr_data[0][0]
#         pr_w_graph = pr_data[0][1]
#         logger.info(prefix + ' #Dets: {0}, #GT: {1} TP: {2} FP: {3} = {4} wrong + {5} bkgnd  Recall: {6:.3f} AP: {7:.3f}'.format(d, nGT, \
#                      np.int(tot_tp), np.int(tot_fp),np.int(tot_fp_wrong), np.int(tot_fp_bkgnd), recall,ap))
#         return
#
#     def save_stats(self, store_stats_fname):
#
#         np.savez(store_stats_fname, stats=[self.sc, self.tp, self.fp, self.fpw, self.fpb, self.nGT, self.d, self.CM])
#
#     def get_stats(self):
#         stat_array = np.expand_dims(self.sc[:self.d], axis=0)
#         stat_array = np.concatenate((stat_array, np.expand_dims(self.tp[:self.d], axis=0)), axis=0)
#         stat_array = np.concatenate((stat_array, np.expand_dims(self.fp[:self.d], axis=0)), axis=0)
#         stat_array = np.concatenate((stat_array, np.expand_dims(self.fpw[:self.d], axis=0)), axis=0)
#         stat_array = np.concatenate((stat_array, np.expand_dims(self.fpb[:self.d], axis=0)), axis=0)
#         stat_array = stat_array.astype(np.float16)
#         return [stat_array, self.nGT, self.d, self.CM]  # [, , self.fp[:self.d], self.fpw[:self.d], self.fpb[:self.d], self.nGT, self.d, self.img_recAtK]

class ObjDetStats():
    def __init__(self,gt_roidb_fname, cat_ords, logger):
        # Ground Truth
        with open(gt_roidb_fname, 'rb') as fid:
            self.roidb = cPickle.load(fid) # TODO: all the object filtering (size, valid_objects, BG_scores) is done offline in roidb.

        # reindex using image names
        self.roidb_ni = {}
        for entry in self.roidb:
            image_path = entry['image']
            im_path, im_name = os.path.split(image_path)
            self.roidb_ni[im_name] = entry

        self.cat_ords =cat_ords
        self.logger = logger# configure_logging(logger_fname)

    # def img_dets_CSV_2_A(self, csv_fname, name_field=False):
    #     # convets detections format from C to B. See the list of formats in RepMetDrive.py
    #     q_dets = [ np.zeros((0,5)) for ord in self.cat_ords]
    #     with open(csv_fname, 'r') as fid:
    #         dets_lines = [x.strip() for x in fid.readlines()]
    #     for det in dets_lines:
    #         fields = det.split(';')
    #         cat_name = fields[0]
    #         cat_ord = int(fields[6])
    #         cat_idx = np.where(np.asarray(self.cat_ords) == cat_ord)[0][0]
    #         det_box = np.expand_dims(np.array([int(fields[2]),int(fields[3]),int(fields[4]),int(fields[5]),float(fields[1])]),axis=0)
    #         if len(q_dets[cat_idx])==0:
    #             q_dets[cat_idx] =det_box
    #         else:
    #             q_dets[cat_idx] = np.concatenate((q_dets[cat_idx],det_box),axis=0)
    #     return q_dets

    # def compute_base_stats_episodes(self,dets_root,Nclasses=1): # Nclasses needed for confusion matrix initialization
    #     ovthresh = 0.5
    #     perf_stats = PerfStats(Nclasses)
    #     for ROOT, DIR, FILES in os.walk(dets_root):
    #         for epi_root in DIR:
    #             fields = epi_root.split('_')
    #             epi_num = int(fields[1])
    #             for ROOT_, DIR_, FILES_ in os.walk(os.path.join(dets_root,epi_root)):
    #                 # fetch detctions in this image in the format of q_dets
    #                 for detsName in FILES_:
    #                     imgName = detName_2_imgName(detsName)
    #                     q_dets,cat_names = self.img_dets_CSV_2_A(os.path.join(dets_root, epi_root,detsName),self.cat_ords)
    #                     entry = self.roidb_ni[imgName]
    #                     # then run
    #                     perf_stats.comp_epi_stats_m(q_dets, entry['boxes'], entry['gt_classes'], self.ord2name.keys(), ovthresh)
    #                 perf_stats.print_perf(self.logger, prefix='{0}'.format(epi_root))

    def print_bboxes(self,dets_folder,query_images):
        import cv2
        from utils.show_boxes import show_dets_gt_boxes
        for detsName in os.listdir(dets_folder):
            if detsName[-4:]=='.txt':
                detsPath = os.path.join(dets_folder,detsName)
                imgName = detName_2_imgName(detsName)
                entry = self.roidb_ni[imgName]
                imgName = detName_2_imgName(detsName)
                for q_im_path in query_images:
                    _,q_im_name = os.path.split(q_im_path)
                    if q_im_name==imgName:
                        im = cv2.imread(q_im_path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        dvisName = imgName_2_dvisName(imgName)
                        q_dets,cat_names = img_dets_CSV_2_A(detsPath,self.cat_ords)
                        show_dets_gt_boxes(im, q_dets,cat_names, entry['boxes'], entry['gt_names'], scale=1.0, FS=12, LW=1.5,
                                           save_file_path=os.path.join(dets_folder, dvisName))
                        break

    # def add_base_stats(self,dets_folder,perf_stats,Nclasses=1): # Nclasses needed for confusion matrix initialization
    #     ovthresh = 0.5
    #     # if perf_stats is None:
    #     #     perf_stats = PerfStats_internal(Nclasses)
    #
    #     for detsName in os.listdir(dets_folder):
    #         if detsName[-4:]=='.txt':
    #             q_dets,cat_names = img_dets_CSV_2_A(os.path.join(dets_folder,detsName), self.cat_ords)
    #             imgName = detName_2_imgName(detsName)
    #             entry = self.roidb_ni[imgName]
    #             # then run
    #             perf_stats.comp_epi_stats_m(q_dets, entry['boxes'], entry['gt_classes'],self.cat_ords, ovthresh)
    #     perf_stats.print_perf(self.logger, prefix='_')
    #     return perf_stats

    def add_base_stats_inDomain(self, dets_folder, perf_stats, Nclasses=1, ovthresh = 0.5):  # Nclasses needed for confusion matrix initialization

        # if perf_stats is None:
        #     perf_stats = PerfStats_internal(Nclasses)

        for detsName in os.listdir(dets_folder):
            if detsName[-4:] == '.txt':
                q_dets, cat_names = img_dets_CSV_2_A(os.path.join(dets_folder, detsName), self.cat_ords)
                imgName = detName_2_imgName(detsName)
                entry = self.roidb_ni[imgName]
                # then run

                gt_boxes_test = []
                gt_classes_test = []
                for gt_box, gt_class in zip( entry['boxes'], entry['gt_classes']):
                    if gt_class in self.cat_ords:
                        gt_boxes_test += [gt_box]
                        gt_classes_test += [gt_class]
                gt_classes_test = np.asarray(gt_classes_test)
                gt_boxes_test = np.asarray(gt_boxes_test)
                perf_stats.comp_epi_stats_m(q_dets, gt_boxes_test, gt_classes_test, self.cat_ords, ovthresh)
        perf_stats.print_perf(self.logger, prefix='_')
        return perf_stats

    # def compute_base_stats(self,dets_folder,Nclasses=1): # Nclasses needed for confusion matrix initialization
    #     ovthresh = 0.5
    #     perf_stats = PerfStats_internal(Nclasses)
    #
    #     for detsName in os.listdir(dets_folder):
    #         if detsName[-4:]=='.txt':
    #             q_dets,cat_names = img_dets_CSV_2_A(os.path.join(dets_folder,detsName), self.cat_ords)
    #             imgName = detName_2_imgName(detsName)
    #             entry = self.roidb_ni[imgName]
    #             # then run
    #             perf_stats.comp_epi_stats_m(q_dets, entry['boxes'], entry['gt_classes'],self.cat_ords, ovthresh)
    #     perf_stats.print_perf(self.logger, prefix='_')


        # for ROOT, DIR, FILES in os.walk(dets_folder):
        #             # fetch detctions in this image in the format of q_dets
        #         for detsName in FILES: