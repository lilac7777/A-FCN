# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
import scipy.io as sio
class ROIAlignLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    
    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        bottom_shape = bottom[0].shape
        idx = 0
        layer_params = yaml.load(self.param_str)

        self.h = layer_params['pooled_h']
        self.w = layer_params['pooled_w']
        self.spatial_scale = layer_params['spatial_scale']
        top[idx].reshape(bottom_shape[0],bottom_shape[1],self.h,self.w)
        self.grad_map = np.zeros((bottom_shape[0],self.h,self.w,8))
        

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        bottom_shape = bottom[0].shape
        top[0].reshape(bottom_shape[0],bottom_shape[1],self.h,self.w)
        top_data = np.zeros((bottom_shape[0],bottom_shape[1],self.h,self.w))
        bottom_data = bottom[0].data
        for n in xrange(bottom_shape[0]):
          bottom_rois = bottom[1].data[n,:]
          roi_height = (bottom[1].data[n,4] - bottom[1].data[n,2])*self.spatial_scale
          roi_width = (bottom[1].data[n,3] - bottom[1].data[n,1])*self.spatial_scale
          pic_w_start = bottom_rois[1] * self.spatial_scale
          pic_h_start = bottom_rois[2] * self.spatial_scale
          for i in xrange(self.h):
            pic_h = pic_h_start + (i + 0.0)*roi_height/self.h
            floor_pic_h = max(0, np.floor(pic_h))
            ceil_pic_h = min(bottom_shape[2],np.ceil(pic_h))
            u = pic_h - floor_pic_h
  
            for j in xrange(self.w):
              pic_w = pic_w_start + (j + 0.0)*roi_width/self.w
              floor_pic_w = max(0, np.floor(pic_w))
              ceil_pic_w = min(bottom_shape[3],np.ceil(pic_w))
              v = pic_w - floor_pic_w
              self.grad_map[n,i,j,0] = floor_pic_h
              self.grad_map[n,i,j,1] = ceil_pic_h
              self.grad_map[n,i,j,2] = floor_pic_w
              self.grad_map[n,i,j,3] = ceil_pic_w
              self.grad_map[n,i,j,4] = u
              self.grad_map[n,i,j,5] = 1 - u
              self.grad_map[n,i,j,6] = v
              self.grad_map[n,i,j,7] = 1 - v
              for c in xrange(bottom_shape[1]):
                  top_data[n,c,i,j] = (1 - u)*(1 - v)*bottom_data[n,c,floor_pic_h,floor_pic_w] + (1 - u)*(v)*bottom_data[n,c,floor_pic_h,ceil_pic_w]+(u)*(1 - v)*bottom_data[n,c,ceil_pic_h,floor_pic_w] + u*v*bottom_data[n,c,ceil_pic_h,ceil_pic_w]
        top[0].data[...] = top_data
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        top_shape = top[0].shape
        bottom_shape = bottom[0].shape
        bottom_diff = np.zeros((bottom_shape[0],bottom_shape[1],bottom_shape[2],bottom_shape[3]))
        for n in xrange(bottom_shape[0]):
          bottom_rois = bottom[1].data[n,:]
          roi_height = (bottom[1].data[n,4] - bottom[1].data[n,2])*self.spatial_scale
          roi_width = (bottom[1].data[n,3] - bottom[1].data[n,1])*self.spatial_scale
          pic_w_start = bottom_rois[1] * self.spatial_scale
          pic_h_start = bottom_rois[2] * self.spatial_scale
          for i in xrange(self.h):
            pic_h = pic_h_start + (i + 0.0)*roi_height/self.h
            floor_pic_h = max(0, np.floor(pic_h))
            ceil_pic_h = min(bottom_shape[2],np.ceil(pic_h))
            u = pic_h - floor_pic_h
            for j in xrange(self.w):
              pic_w = pic_w_start + (j + 0.0)*roi_width/self.w
              floor_pic_w = max(0, np.floor(pic_w))
              ceil_pic_w = min(bottom_shape[3],np.ceil(pic_w))
              v = pic_w - floor_pic_w
              for c in xrange(bottom_shape[1]):
                  bottom_diff[n,c,floor_pic_h,floor_pic_w] += top[0].diff[n,c,i,j]*(1 - u)*(1 - v)
                  bottom_diff[n,c,floor_pic_h,ceil_pic_w] += top[0].diff[n,c,i,j]*(1 - u)*v
                  bottom_diff[n,c,ceil_pic_h,floor_pic_w] += top[0].diff[n,c,i,j]*u*(1 - v)
                  bottom_diff[n,c,ceil_pic_h,ceil_pic_w] += top[0].diff[n,c,i,j]*u*v
        bottom[0].diff[...] = bottom_diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

