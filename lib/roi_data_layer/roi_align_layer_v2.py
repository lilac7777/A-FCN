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
        bottom_roi_shape =  bottom[1].shape
        #print bottom_shape[0],bottom_shape[1],bottom_shape[2],bottom_shape[3]
        top[0].reshape(bottom_shape[0],bottom_shape[1],self.h,self.w)
        top_data = np.zeros((bottom_roi_shape[0],bottom_shape[1],self.h,self.w))
        bottom_data = bottom[0].data
        #print bottom_roi_shape[0],bottom_roi_shape[1],bottom_roi_shape[2],bottom_roi_shape[3],'back'
        for n in xrange(bottom[1].data.shape[0]):
          bottom_rois = np.array(bottom[1].data[n,:]).reshape((5,))
          #print self.spatial_scale,bottom_rois[1]
          roi_height = (bottom_rois[4] - bottom_rois[2])*self.spatial_scale
          roi_width = (bottom_rois[3] - bottom_rois[1])*self.spatial_scale
          pic_w_start = bottom_rois[1] * self.spatial_scale
          pic_h_start = bottom_rois[2] * self.spatial_scale
          for i in xrange(self.h):
            pic_h = pic_h_start + (float(i))*roi_height/self.h
            
            #print pic_h_start,i,roi_height,pic_h,self.h,'sss'
            floor_pic_h = max(0, np.floor(pic_h))
            ceil_pic_h = min(bottom_shape[2]-1,np.ceil(pic_h))
            #print pic_h
            #print bottom_shape[2],np.ceil(pic_h)
            u = pic_h - floor_pic_h
  
            for j in xrange(self.w):
              pic_w = pic_w_start + (j + 0.0)*roi_width/self.w
              floor_pic_w = max(0, np.floor(pic_w))
              ceil_pic_w = min(bottom_shape[3]-1,np.ceil(pic_w))
              v = pic_w - floor_pic_w
              for c in xrange(bottom_shape[1]):
                  #print bottom_rois.shape
                  #print ceil_pic_h
                  #print ceil_pic_w
                  #print floor_pic_h
                  #print floor_pic_w
                  top_data[n,c,i,j] = (1 - u)*(1 - v)*bottom_data[0,c,floor_pic_h,floor_pic_w] + (1 - u)*(v)*bottom_data[0,c,floor_pic_h,ceil_pic_w]+(u)*(1 - v)*bottom_data[0,c,ceil_pic_h,floor_pic_w] + u*v*bottom_data[0,c,ceil_pic_h,ceil_pic_w]
        top[0].reshape(bottom_roi_shape[0],bottom_shape[1],self.h,self.w)
        top[0].data[...] = top_data
        top_shape = top[0].shape
        #print top_shape[0],top_shape[1],top_shape[2],top_shape[3]

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        top_shape = top[0].shape
        bottom_shape = bottom[0].shape
        print bottom_shape[0],bottom_shape[1],bottom_shape[2],bottom_shape[3],'back'
        bottom_diff = np.zeros((bottom_shape[0],bottom_shape[1],bottom_shape[2],bottom_shape[3]))
        for n in xrange(bottom[1].data.shape[0]):
          bottom_rois = np.array(bottom[1].data[n,:]).reshape((5,))
          roi_height = (bottom_rois[4] - bottom_rois[2])*self.spatial_scale
          roi_width = (bottom_rois[3] - bottom_rois[1])*self.spatial_scale
          pic_w_start = bottom_rois[1] * self.spatial_scale
          pic_h_start = bottom_rois[2] * self.spatial_scale
          pic_w_end = bottom_rois[3] * self.spatial_scale
          pic_h_end = bottom_rois[4] * self.spatial_scale
          for i in xrange(round(pic_h_start),round( pic_h_end)):
            roi_h = (i - pic_h_start+ 0.0)/roi_height*self.h
            floor_roi_h = max(0, np.floor(roi_h))
            ceil_roi_h = min(self.h-1,np.ceil(roi_h))
            u = roi_h - floor_roi_h
            for j in xrange(round(pic_w_start),round( pic_w_end)):
              roi_w = (j - pic_w_start+ 0.0)/roi_width*self.w
              floor_roi_w = max(0, np.floor(roi_w))
              ceil_roi_w = min(self.w-1,np.ceil(roi_w))
              v = roi_w - floor_roi_w
              for c in xrange(bottom_shape[1]):
                  bottom_diff[0,c,i,j] += (1 - u)*(1 - v)*top_diff[n,c,floor_roi_h,floor_roi_w] + (1 - u)*(v)*top_diff[n,c,floor_roi_h,ceil_roi_w]+(u)*(1 - v)*top_diff[n,c,ceil_roi_h,floor_roi_w] + u*v*top_diff[n,c,ceil_roi_h,ceil_roi_w]
        bottom[0].diff[...] = bottom_diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

