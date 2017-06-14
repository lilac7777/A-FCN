import sys
sys.path.append('/net/wujial/py-R-FCN/caffe/python')
sys.path.append('/net/wujial/py-R-FCN/lib')
import caffe
import numpy as np
import scipy.io as sio
from plot_fig import *
from fast_rcnn.test import _get_blobs
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import cv2
from fast_rcnn.config import cfg
cfg.TEST.HAS_RPN = True
cfg.TRAIN.SCALE = (600,)
cfg.TEST.SCALE = (600,)
model_name = 'residual_pos_neg_attention_12'
#model_name = 'rfcn_alt_opt_5step_ohem'
im = cv2.imread('/net/wujial/py-R-FCN/data/demo/006177.jpg')
rpn_proto = "/net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/rfcn_alt_opt_5step_ohem/rpn_test.pt"
rpn_model = "/net/wujial/py-R-FCN/output/rfcn_alt_opt_5step_ohem/voc_2007_trainval/stage1_rpn_final.caffemodel"
prototxt = '/net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/' + model_name + '/soft_rfcn_test.pt'
model = '/net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/' + model_name + '/resnet50_rfcn_mask_ohem_iter_75000.caffemodel'
#model = '/net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/' + model_name + '/stage1_mask_rfcn_final.caffemodel'
#model = "/net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/rfcn_alt_opt_5step_ohem/resnet50_rfcn_ohem_iter_120000.caffemodel"

caffe.set_mode_gpu()
caffe.set_device(2)
rfcn_net = caffe.Net(rpn_proto,rpn_model, caffe.TEST)

blobs, im_scales = _get_blobs(im, None)
im_blob = blobs['data']
blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
rfcn_net.blobs['data'].reshape(*(blobs['data'].shape))
rfcn_net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
blobs_out = rfcn_net.forward(**forward_kwargs)
rois = rfcn_net.blobs['rois'].data.copy()

boxes = rois[10:30, 1:5] / im_scales[0]

cfg.TEST.HAS_RPN = False
blobs, im_scales = _get_blobs(im, boxes)
rfcn_net = caffe.Net( prototxt , model, caffe.TEST)
#blobs, im_scales = _get_blobs(im, pred_boxes)
v = np.array([1, 1e3, 1e6, 1e9, 1e12])
hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
_, index, inv_index = np.unique(hashes, return_index=True,return_inverse=True)
caffe.set_mode_gpu()
caffe.set_device(2)

rfcn_net.blobs['data'].reshape(*(blobs['data'].shape))
rfcn_net.blobs['rois'].reshape(*(blobs['rois'].shape))
forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)


blobs_out = rfcn_net.forward(**forward_kwargs)
scores = blobs_out['cls_prob']
box_deltas = blobs_out['bbox_pred']
bbox_means = [0.0, 0.0, 0.0, 0.0, 1.03960042775271e-10,0.00622199373803706,0.0207805908339361,0.0524860248101128]
bbox_stds = [0.0 ,0.0, 0.0, 0.0, 0.131444678954748,0.125309184804088,0.249703604170591,0.216150527133179]
box_deltas = box_deltas * (np.repeat(bbox_stds, box_deltas.shape[0]).reshape(box_deltas.shape[0], 8 )) + np.repeat(bbox_means, box_deltas.shape[0]).reshape(box_deltas.shape[0], 8 )

#box_deltas = box_deltas[inv_index,:]
pred_boxes = bbox_transform_inv(blobs['rois'][:,1:5] / im_scales[0], box_deltas)
pred_boxes = clip_boxes(pred_boxes, im.shape)
#scores = scores[inv_index, :]
#pred_boxes = pred_boxes[inv_index, :]

data = rfcn_net.blobs['data'].data
#mask = rfcn_net.blobs['Sig_pos_mask'].data
#soft_roi_cls = rfcn_net.blobs['roi_pool_pos_cls'].data


cls = scores.argmax(axis = 1)
ind = np.where(cls > 0)[0][0]
ind1 = np.where(cls > 0)[0][1]
ind2 = np.where(cls > 0)[0][2]
imim = blobs['data'][0,0,:,:]
inds = np.where(cls > 0)[0]
tmp = scores[inds,cls[inds]]
tmptmp = np.argsort(tmp)

pred_boxes = pred_boxes * im_scales[0]
print pred_boxes[inds[tmptmp[-1]],4:8]
print pred_boxes[inds[tmptmp[-2]],4:8]
print pred_boxes[inds[tmptmp[-3]],4:8]
for i in xrange(pred_boxes.shape[0]):
  save_imagesc(imim ,detection =pred_boxes[inds[tmptmp[-1]],4:8].reshape((1,4)),name= 'img0')
  save_imagesc(imim ,detection =pred_boxes[inds[tmptmp[-2]],4:8].reshape((1,4)),name= 'img1')
  save_imagesc(imim ,detection =pred_boxes[inds[tmptmp[-3]],4:8].reshape((1,4)),name= 'img2')

#for i in range(25):
#  save_imagesc(soft_roi_cls[ind,0*25 + i,:,:],'fig/background_%d'%i)
#  save_imagesc(soft_roi_cls[ind,18*25+ i,:,:],'fig/foreground_%d'%i)
#  save_imagesc(mask[0, i,:,:],None,'fig/mask_%d'%i)

saveto = 'test_out.mat'
netdata = dict()
netdata['data'] = data
netdata['im_scales'] = im_scales[0]
#netdata['output'] = net.blobs['proposal'].data
#netdata['mask'] = mask
netdata['scores'] = scores
netdata['pred_boxes'] = pred_boxes
#netdata['rfcn_neg_cls_soft'] = rfcn_net.blobs['rfcn_neg_cls_soft'].data
netdata['rfcn_pos_cls_soft'] = rfcn_net.blobs['rfcn_pos_cls_soft'].data
netdata['data'] = rfcn_net.blobs['data'].data
netdata['Sig_pos_mask'] = rfcn_net.blobs['Sig_pos_mask'].data
netdata['Sig_neg_mask'] = rfcn_net.blobs['Sig_neg_mask'].data
netdata['cls_score'] = rfcn_net.blobs['cls_score'].data
netdata['rois'] = rfcn_net.blobs['rois'].data
netdata['deterministic'] = rfcn_net.blobs['deterministic'].data
netdata['roi_pool_pos_mask'] = rfcn_net.blobs['roi_pool_pos_mask'].data
netdata['roi_pool_neg_mask'] = rfcn_net.blobs['roi_pool_neg_mask'].data
sio.savemat(saveto,netdata)