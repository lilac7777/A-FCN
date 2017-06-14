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

proto = "/net/wujial/py-R-FCN/test_cross_conv_layer.prototxt"
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(proto, caffe.TEST)
weights = np.random.randn(512,128,3,3)
data = np.random.randn(100,128,50,50)
net.blobs['data'].data[...]  = data
net.blobs['weights'].data[...]  = weights
out = net.forward()


