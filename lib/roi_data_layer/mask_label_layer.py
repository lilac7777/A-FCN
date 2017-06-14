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
import numpy as np
import yaml
from multiprocessing import Process, Queue

class MaskLabelLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._neg = 0
        if layer_params.has_key('neg'):
            self._neg = layer_params['neg']

        self._name_to_top_map = {}
        
        shape = bottom[0].data.shape
        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(1, cfg.TRAIN.mask_num , shape[2], shape[3])
        self._name_to_top_map['mask_label'] = idx
        idx += 1
        
        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        
        cls_rfcn = bottom[0].data#(n, c*k, w , h)        
        roi_label = bottom[1].data #(n, 1)
        shape = bottom[0].data.shape
        cls_rfcn_inds = cls_rfcn.reshape((shape[0], shape[1]/cfg.TRAIN.mask_num, cfg.TRAIN.mask_num,shape[2],shape[3])).argmax(axis = 1)
        mask_label = np.zeros((shape[0],cfg.TRAIN.mask_num,shape[2],shape[3]))
        for i in xrange(shape[0]):
            #print cls_rfcn_inds[i,:,:,:].shape,roi_label[i].shape
            if self._neg == 1:
                mask_label[i,:,:,:]= (cls_rfcn_inds[i,:,:,:]  == 0).astype('float32')
            if self._neg == 0:
                mask_label[i,:,:,:]= (cls_rfcn_inds[i,:,:,:]  == roi_label[i] - 1).astype('float32')
            
            
        #print cls_rfcn_inds[0,0,:,:],mask_label[0,0,:,:]
        
        top[0].reshape(shape[0],cfg.TRAIN.mask_num,shape[2],shape[3])
        top[0].data[...] = mask_label
        ##print mask_label.shape
        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

