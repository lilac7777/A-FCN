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
proto = "/net/wujial/py-R-FCN/test_layer_solver.prototxt"
caffe.set_mode_gpu()
caffe.set_device(3)
solver = caffe.SGDSolver(proto)
data = np.random.randn(100,525,7,7)
mask = np.random.randn(100,25,7,7)
solver.net.blobs['data'].data[...]  = data
solver.net.blobs['mask'].data[...]  = mask
out = solver.step(1)


test = solver.net.blobs['cls_score_test'].data
lilac = solver.net.blobs['cls_score'].data
print (test-lilac).sum()
