import cPickle
import mxnet as mx
from utils.load_model import load_param

def ListList2ndarray(LList):
    dim_in = len(LList)
    dim_out = len(LList[0])
    weight = mx.ndarray.zeros([dim_in, dim_out, ])
    for i, inp in enumerate(LList):
        weight[i, :] = mx.nd.array(inp)
    weight = weight.T
    return weight

def load_Prot_rep(config, vanilla_model, embed_weights_fname):

    epoch = 0
    arg_params, aux_params = load_param(vanilla_model, epoch, process=True)
    with open(embed_weights_fname, 'rb') as fid:
        model_data = cPickle.load(fid)
    layer =  model_data['feats1']
    weight = ListList2ndarray(layer[0])
    bias = mx.nd.array(layer[1])

    arg_params['embed_dense_1_weight'] = weight
    arg_params['embed_dense_1_bias'] = bias

    layer = model_data['batch_normalization_3']
    moving_mean = mx.nd.array(layer[2])  # mx.nd.array(np.array(layer[0]))
    moving_var = mx.nd.array(layer[3])
    gamma = mx.nd.array(layer[0])
    beta = mx.nd.array(layer[1])
    fcn = '1'
    aux_params['embed_batchNorm_' + fcn + '_moving_mean'] = moving_mean
    aux_params['embed_batchNorm_' + fcn + '_moving_var'] = moving_var
    arg_params['embed_batchNorm_' + fcn + '_gamma'] = gamma
    arg_params['embed_batchNorm_' + fcn + '_beta'] = beta

    return arg_params, aux_params

def load_fpn_rep_emb(config, vanilla_model, embed_weights_fname,epoch=0):
    # -----------------------------------------------------------------------------
    # load vanilla fpn weights
    arg_params, aux_params = load_param(vanilla_model, epoch, process=True)
    # load embedding weights. source: 'dev_vanilla_embed.py'

    with open(embed_weights_fname, 'rb') as fid:
        model_data = cPickle.load(fid)
    for fcn in ['1', '2', '3']:
        layer = model_data['dense_' + fcn]
        weight = ListList2ndarray(layer[0])
        bias = mx.nd.array(layer[1])
        arg_params['embed_dense_' + fcn + '_weight'] = weight
        arg_params['embed_dense_' + fcn + '_bias'] = bias

    # BatchNormalization has four parameters: Running mean, running standard deviation, gamma and beta
    for fcn in ['1', '2']:  # fcn = '1'
        layer = model_data['batchNorm_' + fcn]
        moving_mean = mx.nd.array(layer[2])  # mx.nd.array(np.array(layer[0]))
        moving_var = mx.nd.array(layer[3])
        gamma = mx.nd.array(layer[0])
        beta = mx.nd.array(layer[1])
        aux_params['embed_batchNorm_' + fcn + '_moving_mean'] = moving_mean
        aux_params['embed_batchNorm_' + fcn + '_moving_var'] = moving_var
        arg_params['embed_batchNorm_' + fcn + '_gamma'] = gamma
        arg_params['embed_batchNorm_' + fcn + '_beta'] = beta

    return arg_params, aux_params

def load_scalar_pred_subnet(config, vanilla_model, embed_weights_fname,epoch=0):
    #load vanilla fpn weights
    arg_params, aux_params = load_param(vanilla_model, epoch, process=True)
    with open(embed_weights_fname, 'rb') as fid:
        model_data = cPickle.load(fid)
    for fcn in ['1', '2', '3']:
        layer = model_data['dense' + fcn]
        weight = ListList2ndarray(layer[0])
        bias = mx.nd.array(layer[1])
        arg_params['fc_score_hist_' + fcn + '_weight'] = weight
        arg_params['fc_score_hist_' + fcn + '_bias'] = bias

    return arg_params, aux_params

def load_bgw_scov_subnet(config, vanilla_model, embed_weights_fname,epoch=0):
    #load vanilla fpn weights
    arg_params, aux_params = load_param(vanilla_model, epoch, process=True)
    with open(embed_weights_fname, 'rb') as fid:
        model_data = cPickle.load(fid)
    for fcn in ['1', '2', '3']:
        layer = model_data['dense' + fcn]
        weight = ListList2ndarray(layer[0])
        bias = mx.nd.array(layer[1])
        arg_params['scov_dense_' + fcn + '_weight'] = weight
        arg_params['scov_dense_' + fcn + '_bias'] = bias

    return arg_params, aux_params

def load_bgw_pred_subnet(config, vanilla_model, embed_weights_fname,epoch=0):
    #load vanilla fpn weights
    arg_params, aux_params = load_param(vanilla_model, epoch, process=True)
    with open(embed_weights_fname, 'rb') as fid:
        model_data = cPickle.load(fid)
    for fcn in ['1', '2', '3']:
        layer = model_data['dense' + fcn]
        weight = ListList2ndarray(layer[0])
        bias = mx.nd.array(layer[1])
        arg_params['bgw_pred_dense_' + fcn + '_weight'] = weight
        arg_params['bgw_pred_dense_' + fcn + '_bias'] = bias

    return arg_params, aux_params

def load_TLrep_emb(config, vanilla_model, embed_weights_fname,epoch=0):
    # load vanilla fpn weights
    arg_params, aux_params = load_param(vanilla_model, epoch, process=True)
    # load embedding weights. source: 'dev_vanilla_embed.py'

    with open(embed_weights_fname, 'rb') as fid:
        model_data = cPickle.load(fid)
    for fcn in ['1', '2', '3']:
        layer = model_data['dense_' + fcn]
        weight = ListList2ndarray(layer[0])
        bias = mx.nd.array(layer[1])
        arg_params['embed_dense_' + fcn + '_weight'] = weight
        arg_params['embed_dense_' + fcn + '_bias'] = bias

    return arg_params, aux_params

# def load_a_model_reset_cls(config,model_case=0):
#     import numpy as np
#     import mxnet as mx
#     if model_case == 10: # vanilla pretrained on COCO
#         epoch = 0
#         def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'
#         arg_params, aux_params = load_param(def_model, epoch, process=True)
#
#         arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_params['cls_score_weight'].shape)
#         arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_params['cls_score_bias'].shape)
#         arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_params['bbox_pred_weight'].shape)
#         arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_params['bbox_pred_bias'].shape)

#        arg_params['cls_score_weight'] = 0.01*np.random.random_sample(arg_params['cls_score_weight'].shape)# np.random(0, 0.01, shape=arg_params['cls_score_weight'].shape)

    #return arg_params, aux_params

def load_a_model(config,model_case=0):

    # load a FPN reps model for yaml 8, epoch = 15, prep representative weights
    if model_case == 'RepMet_inloc':
        def_model = '/dccstor/jsdata1/dev/RepMet_project/RepMet_CVPR19/data/Imagenet_LOC/fpn_pascal_imagenet'
        epoch = 15
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case =='RM_fpn_JES_poc19':
        epoch = 19
        def_model = '/dccstor/jsdata1/dev/RepMet/output/fpn/JES_poc/cfg_01b/poc_train/fpn_JES_poc'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case =='15_van':
        epoch = 13
        def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/voc_imagenet_det/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15_Van/2007_trainval+2012_trainval_train:DET/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case == 'FPN_DCN': # 10: # vanilla pretrained on COCO
        epoch = 0
        def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'
        arg_params, aux_params = load_param(def_model, epoch, process=True)
        #return arg_params, aux_params
    #from config.config import config, update_config

    if model_case == '19': # 303 Vanilla fpn-dcn trained on Pascal + IN-LOC
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        epoch = 11
        arg_params, aux_params = load_param(vanilla_model, epoch, process=True)

    if model_case == 19:
        epoch = 22
        def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/coco/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_16/train2014_valminusminival2014/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case == 20:
        epoch = 20
        def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/coco/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_16_CmP/train2014+valminusminival2014/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case == 21:
        epoch = 10
        def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/coco/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_16_CmP_Van/train2014+valminusminival2014/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case ==99:
        def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/cfg_2/train_loc/RepMet'
        epoch = 4
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case == 100:
        vanilla_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'  # vanilla_model
        embed_weights_fname = '/dccstor/jsdata1/data/magnet/models_01/model_epoch49_data.pkl'
        arg_params, aux_params = load_fpn_rep_emb(config,vanilla_model,embed_weights_fname)

    if model_case == 101:
        vanilla_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'  # vanilla_model
        embed_weights_fname = '/dccstor/jsdata1/data/magnet/models_27b/model_epoch65_data.pkl'
        arg_params, aux_params = load_fpn_rep_emb(config,vanilla_model,embed_weights_fname)
    # SimpleTriplet model 3_1

    if model_case == 102:
        vanilla_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'  # vanilla_model
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/model_weights_3_1.pkl'
        arg_params, aux_params = load_fpn_rep_emb(config,vanilla_model,embed_weights_fname)
    # Prototypical embeding

    if model_case == 103:
        vanilla_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'  # vanilla_model
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/Proto_model_1/model_proto_1_weights.pkl'
        arg_params, aux_params = load_Prot_rep(config,vanilla_model,embed_weights_fname)

    if model_case == 104:
        vanilla_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'  # vanilla_model
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/TLrep_model_weights_1.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname)

    if model_case == 114:
        epoch = 11
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/TLrep_model_weights_1.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 115:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/TLrep_model_weights_6_03.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 116:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/TLrep_model_weights_6_M03.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 1161:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/TLrep_model_weights_6_M04a_256_0.05.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 1162:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/ExtEmb_7_fc1.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 1163:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/ExtEmb_7_fc2.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 1164:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/ExtEmb_7_fc3.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 1164:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/ExtEmb_7_fc3.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)




    if model_case == 117:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/TLrep_model_weights_6_M1_03.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 118:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/dev/josephs-magnet/DL_embed/proj_bgEmb_01/TLrep_model_weights_6_M2_03.pkl'
        arg_params, aux_params = load_TLrep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 119: # Vanilla fpn-dcn trained on Pascal + IN-LOC
        epoch=11
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/data/magnet/models_01/model_epoch49_data.pkl'
        arg_params, aux_params = load_fpn_rep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 120:   # Vanilla fpn-dcn trained on Pascal + IN-LOC   +  embedding
        epoch=11
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'
        embed_weights_fname = '/dccstor/jsdata1/data/magnet/models_01/model_epoch49_data.pkl'
        arg_params, aux_params = load_fpn_rep_emb(config,vanilla_model,embed_weights_fname,epoch)
        return arg_params, aux_params

    if model_case == 129: # Vanilla fpn-dcn trained on Pascal + IN-LOC, epoch 15
        epoch=15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/data/magnet/models_01/model_epoch49_data.pkl'
        arg_params, aux_params = load_fpn_rep_emb(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 130: # Vanilla fpn-dcn trained on Pascal + IN-LOC, epoch 15, no representative weights
        epoch=15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        embed_weights_fname = '/dccstor/jsdata1/data/magnet/models_01/model_epoch49_data.pkl'
        arg_params, aux_params = load_fpn_rep_emb(config,vanilla_model,embed_weights_fname,epoch)
        return arg_params, aux_params


    if model_case == 140:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'
        #embed_weights_fname = '/dccstor/jsdata1/data/inloc_feats_cfg8/model_1_data.pkl'
        embed_weights_fname = '/dccstor/jsdata1/data/inloc_feats_cfg8_2/model_1_data.pkl'
        arg_params, aux_params = load_bgw_pred_subnet(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 141:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'
        #embed_weights_fname = '/dccstor/jsdata1/data/inloc_feats_cfg8/model_1_data.pkl'
        #embed_weights_fname = '/dccstor/jsdata1/data/inloc_feats_cfg8_2/model_2_data.pkl'
        embed_weights_fname = '/dccstor/jsdata1/data/inloc_feats_cfg8_2/model_3_data.pkl'
        arg_params, aux_params = load_bgw_pred_subnet(config,vanilla_model,embed_weights_fname,epoch)

    if model_case == 142:
        epoch = 15
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'
        embed_weights_fname = '/dccstor/jsdata1/data/inloc_feats_cfg8/model_1_data.pkl'
        arg_params, aux_params = load_bgw_scov_subnet(config,vanilla_model,embed_weights_fname,epoch)



    if model_case == 1200:
        epoch = 15
        def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)
        return arg_params, aux_params

    # load a FPN reps model Imagenet-DET, for yaml 14, epoch = 17
    if model_case == 201:
        epoch = 17
        def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet_det/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_14/2007_trainval_2012_trainval;train:DET/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case == 202:
        epoch = 20
        def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet_det/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_14/2007_trainval_2012_trainval;train:DET/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case == 210:
        epoch = 17
        def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet_det/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15/2007_trainval_2012_trainval;train:DET/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)


    if model_case == 'cfg_15_29': # 211:
        epoch = 29
        def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet_det/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15/2007_trainval_2012_trainval;train:DET/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    if model_case == 2111:
        epoch = 20
        def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet_det/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15/2007_trainval_2012_trainval;train:DET/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)


    if model_case == 2112:
        epoch = 15
        def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet_det/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15/2007_trainval_2012_trainval;train:DET/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)

    # load conf 20 model
    if model_case == 220:
        model_name = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'
        epoch = 3
        arg_params, aux_params = load_param(model_name, epoch, process=True)
        return arg_params, aux_params

    if model_case == 221:
        model_name = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_hist/2007_trainval+2012_trainval_train_loc/fpn_pascal_imagenet'
        epoch = 1
        arg_params, aux_params = load_param(model_name, epoch, process=True)
        return arg_params, aux_params

    if model_case == 2211:
        model_name = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_hist/2007_trainval+2012_trainval_train_loc/fpn_pascal_imagenet'
        subnet_name = '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_eer_score_hist/pred_model_feat_data.pkl'
        epoch = 1
        arg_params, aux_params = load_scalar_pred_subnet(config, model_name,subnet_name, epoch)
        return arg_params, aux_params


    if model_case == 222:
        model_name = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_hist_tr1/2007_trainval+2012_trainval_train_loc/fpn_pascal_imagenet'
        epoch = 8
        arg_params, aux_params = load_param(model_name, epoch, process=True)
        return arg_params, aux_params

    if model_case == 223:
        model_name = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_hist_tr1_w1/2007_trainval+2012_trainval_train_loc/fpn_pascal_imagenet'
        epoch = 8
        arg_params, aux_params = load_param(model_name, epoch, process=True)
        return arg_params, aux_params

    # if model_case  == 230: # cfg8 + subnetwork for bkgd
    #     epoch = 15
    #     def_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'
    #     subnet_name = '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_eer_score_hist/pred_model_feat_data.pkl'
    #     arg_params, aux_params = load_scalar_pred_subnet(config, def_model,subnet_name, epoch)


    # load vanilla FPN model trained on coco
    if model_case == 300:
        vanilla_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/model/fpn_dcn_coco'  # vanilla_model
        arg_params, aux_params = load_param(vanilla_model, 0, process=True)
        return arg_params, aux_params

    # load vanilla FPN model trained on Pascal + IN-LOC
    if model_case == 301:
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        epoch = 15
        arg_params, aux_params = load_param(vanilla_model, epoch, process=True)
        return arg_params, aux_params

    # load vanilla FPN model trained on Pascal + IN-LOC + initialize weights
    if model_case == 302:
        vanilla_model = '/dccstor/leonidka1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19/2007_trainval_2012_trainval;train_loc/fpn_pascal_imagenet'  # vanilla_model trained on Pascal+In-loc101
        epoch = 15
        arg_params, aux_params = load_param(vanilla_model, epoch, process=True)



    if model_case == 400: # RepMet with 1024 features. 21_noClust
        epoch = 18
        def_model = '/dccstor/jsdata1/dev/Deformable-ConvNets/output/fpn/voc_imagenet/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_21_noClust/2007_trainval+2012_trainval_train_loc/fpn_pascal_imagenet'
        arg_params, aux_params = load_param(def_model, epoch, process=True)


    n = config.network.EMBEDDING_DIM * config.network.REPS_PER_CLASS * (config.dataset.NUM_CLASSES - 1)
    arg_params['fc_representatives_weight'] = 1000 * mx.nd.ones((n, 1))
    return arg_params, aux_params

def load_roidb(case = 100):
    if case == 100: # roidb of Imageent-loc train
        imagenet_roidb_fname = '/dccstor/jsdata1/data/voc_inloc/voc_inloc_roidb.pkl'
        with open(imagenet_roidb_fname, 'rb') as fid:
            roidb_data = cPickle.load(fid)
        roidb = roidb_data['roidb']
        classes = roidb_data['classes']
        cls2img = roidb_data['cls2img']

    if case == 101:  # roidb of Imageent-loc val
        roidb_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/data/cache/imagenet_clsloc_val_roidb.pkl'
        with open(roidb_fname, 'rb') as fid:
            roidb_data = cPickle.load(fid)
        roidb = roidb_data['roidb']
        if 'classes' in roidb_data:
            classes = roidb_data['classes']
        else:
            classes = []
        cls2img = roidb_data['cls2img']

    if case == 102: # Imagenet-DET
        roidb_fname = '/dccstor/jsdata1/data/Imagenet_DET/imagenet_det_roidb_corr.pkl'
        with open(roidb_fname, 'rb') as fid:
            roidb_data = cPickle.load(fid)
        roidb = roidb_data['roidb']
        if 'classes' in roidb_data:
            classes = roidb_data['classes']
        else:
            classes = []
        cls2img = roidb_data['cls2img']

    return roidb,classes,cls2img

def config_list(config_id):
    if config_id=='RepMet_inloc':
        cfg_fname = './experiments/cfgs/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8.yaml'

    # if config_id == 'FPN_DCN_test':
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_dcn_rep_noemb.yaml'
    # if config_id == '15_test': #216:  # our model - conf 15 - test
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15_test.yaml'
    # if config_id == '15_Van':
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15_Van.yaml'
    # if config_id == '19_noemb': #103: # Vanilla fpn-dcn trained on Pascal + IN-LOC,
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19_noemb.yaml'
    # if config_id == '19_TLemb': # 101:  # Vanilla fpn-dcn trained on Pascal + IN-LOC, with output emb vector (256)
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19_TLemb.yaml'
    #
    # if config_id == 'fpn_dcn_rep_noemb': #9:
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/resnet_v1_101_voc0712_trainval_fpn_dcn_rep_noemb.yaml'
    # if config_id == 10: # core vanilla model trained on COCO
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_end2end_ohem.yaml'
    # if config_id == 11:  # core vanilla model trained on COCO
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem_featOut.yaml'
    # if config_id == 12: # core vanilla model trained on COCO
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem_FT.yaml'
    # if config_id == 19:  # core vanilla model trained on COCO without Pascal
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_16.yaml'
    # if config_id == 20:  # core vanilla model trained on COCO without Pascal
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_16_CmP.yaml'
    # if config_id == 21:  # core vanilla model trained on COCO without Pascal
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_16_CmP_Van.yaml'
    #
    #
    # if config_id==99:  # Vanilla fpn-dcn trained on Pascal + IN-LOC, with output FC2 feat veactor
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/cfg_2.yaml'
    # if config_id==100:  # Vanilla fpn-dcn trained on Pascal + IN-LOC, with output FC2 feat veactor
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19_FeatOut.yaml'
    # if config_id==102:  # Vanilla fpn-dcn trained on Pascal + IN-LOC, with output emb vector (256)
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19_TLemb1024.yaml'
    # if config_id == 1031: # Vanilla fpn-dcn trained on Pascal + IN-LOC,
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_19_noemb_10rRois.yaml'
    # if config_id==140:  # Vanilla fpn-dcn trained on Pascal + IN-LOC, with output emb vector (256)
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8_bkgd.yaml'
    # if config_id==142:  # Vanilla fpn-dcn trained on Pascal + IN-LOC, with output emb vector (256)
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8_scov.yaml'
    #
    # if config_id==400:  # RepMet 1024 ------------------ 21_noClust
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_21_noClust.yaml'
    #
    # if config_id == 214:  # our model - conf 14
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_14.yaml'
    # if config_id == 215:  # our model - conf 15
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_15.yaml'
    #
    # if config_id==220:  # our model - conf 20
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20.yaml'
    # if config_id == 221:  # our model - conf 20
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_hist.yaml'
    # if config_id == 2211:  # our model - conf 20
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_histE.yaml'
    # if config_id == 222:  # our model - conf 20
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_hist_tr1.yaml'
    # if config_id == 223:  # our model - conf 20
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_20_hist_tr1_w1.yaml'
    # if config_id == 224:  # our model - conf 8
    #     cfg_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/experiments/fpn/cfgs/oneshot/resnet_v1_101_voc0712_trainval_fpn_dcn_oneshot_end2end_ohem_8_bkgd.yaml'

    return cfg_fname

def get_train_objects_fname(case=100):
    if case == 'repmet_indet': # RepMet In-DET:
        train_objects_fname =  '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/a_RepMet/train_objects_RepMet_indet.pkl'
        scores_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fpn/RepMet_indet/BG_scores.pkl'

    if case == 'vanilla_indet': #51:  # Vanilla In-DET:
        scores_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fpn/Vanilla_indet/BG_scores.pkl'
        train_objects_fname =  '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/a_Vanilla/train_objects_nb29_indet.pkl'

    if case == 'repmet_inloc':  # 100 RepMet In-Loc
        scores_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fpn/worst_objects/dev_36/BG_scores.pkl'
        train_objects_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_pascal_imagenet_15/files_train_objects_eer_214_train_hist_2/train_objects_eer_214_train_hist.pkl'

    if case == 'vanilla_inloc':  # 102:  # Vanilla  In-Loc
        scores_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fpn/worst_objects/dev_35/BG_scores.pkl'
        train_objects_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_noemb/train_objects_nb19_214_train_hist.pkl'

    if case == 'ExtEmb': # 103 # TLemb19bM
        train_objects_fname ='/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_TLemb19bM/train_objects_TLemb19bM_214_train.pkl'
        scores_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fpn/TLemb19bM/BG_scores.pkl'

    if case == 99: # cfg2_pre20
        return '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/cfg2_pre20/train_objects_cfg2_pre20_0.pkl'

    if case == 101: # TLemb In-Loc
        return '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_TLemb19rM_hist/train_objects_TLemb19rM_214_train_hist.pkl'




    return train_objects_fname, scores_fname

def cat_list(cat_id):

    if cat_id =='inloc_animals_test': # 214 test Imagenet-LOC categories selected to contain animals and birds.
        cat_list_fname ='./data/Imagenet_LOC/in_domain_categories.txt' #'/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_pascal_imagenet_15/in_domain_categories.txt'

    # if cat_id ==99:
    #     cat_list_fname ='/dccstor/jsdata1/data/inloc_classes_pre20.txt'
    #
    #
    # if cat_id ==200: # 50 out of 214 test Imagenet-LOC categories selected to contain animals and birds. Used in the NIPS paper.
    #     cat_list_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_pascal_imagenet_15/in_domain_categories_50.txt'
    #
    #
    # if cat_id == 101:  # First 101 Imagenet-LOC categories(used for training)
    #     cat_list_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_pascal_imagenet_15/inloc_first101_categories.txt'
    #
    # if cat_id == 102:  # Validation categories 200 man made
    #     cat_list_fname = '/dccstor/jsdata1/dev/Deformable-ConvNets/fewshot_test/fpn_pascal_imagenet_15/inloc_last_200.txt'
    #
    # # Imagenet-DET ------------------------------------------------------------------------
    # # Imagenet-DET 100-100 splits of categories:
    # # split 1: training categories picked to contain the COCO categories
    # # split 2: training categories are the first 100 in the natural order of Imagenet DET'
    #
    # if cat_id == 'indet_split1_train':  # (110) Imagenet-DET, handpicked - train for cfg_15. Containing COCO classes.
    #     cat_list_fname = '/dccstor/jsdata1/data/Imagenet_DET/train_classes.txt'
    # if cat_id == 'indet_split1_test':  # (111) Imagenet-DET, handpicked - test for cfg_15
    #     cat_list_fname = '/dccstor/jsdata1/data/Imagenet_DET/test_classes.txt'
    # # if cat_id == 112:  # (112) Imagenet-DET, handpicked - test 14
    # #     cat_list_fname = '/dccstor/jsdata1/data/Imagenet_DET/imagenet_det_second100_classes.txt','/dccstor/jsdata1/data/imagenet_det_last100.txt'
    # if cat_id == 'indet_split2_train':  # (112) Imagenet-DET, first 100 - train 15
    #     cat_list_fname = '/dccstor/jsdata1/data/imagenet_det_first100.txt'
    # if cat_id == 'indet_split2_test':
    #     cat_list_fname = '/dccstor/jsdata1/data/Imagenet_DET/imagenet_det_second94_classes.txt'
    #
    # if cat_id == 'indet_splitLSTD_test':  # 50 out of Imagenet-DET indet_split1_test
    #     cat_list_fname = '/dccstor/jsdata1/data/Imagenet_DET/test_classes_50.txt'
    #
    # # Pascal VOC
    # if cat_id == 120:
    #     cat_list_fname = '/dccstor/jsdata1/dev/data/VOC_data/VOC_class_list.txt'

    return cat_list_fname

def names_list_to_indices(test_classes_fname,classes):
    import numpy as np
    with open(test_classes_fname, 'r') as fid:
        requested_test_class_names = [x.strip().lower() for x in fid.readlines()]
    #class name to index
    Lclasses = []
    for class_name in classes:
        Lclasses+=[class_name.lower()]
    requested_test_classes = [Lclasses.index(cls_name)+1 for cls_name in requested_test_class_names if cls_name in Lclasses]#
    #requested_test_classes = np.sort(requested_test_classes)
    return requested_test_classes,requested_test_class_names