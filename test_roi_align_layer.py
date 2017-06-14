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

proto = "/net/wujial/py-R-FCN/test_roi_align_layer.prototxt"
caffe.set_mode_cpu()
net = caffe.Net(proto, caffe.TEST)
data = np.random.randn(1,3,60,80)
mask = [0,100,100,300,300]
net.blobs['data'].data[...]  = data
net.blobs['rois'].data[...]  = mask
out = net.forward()


