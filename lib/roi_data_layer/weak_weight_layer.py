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
from plot_fig import *




class WeakWeightLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._num_sample = 10


        self._name_to_top_map = {}
        shape = bottom[0].data.shape

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(1, cfg.TRAIN.mask_num, shape[2], shape[3])
        self._name_to_top_map['weak_in_weight'] = idx
        idx += 1
        top[idx].reshape(1,  cfg.TRAIN.mask_num,shape[2], shape[3])
        self._name_to_top_map['weak_out_weight'] = idx
        idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        mask = bottom[0].data
        shape = bottom[0].data.shape
        
        weights = np.zeros((shape[0],shape[1],shape[2]*shape[3]))
        for i in xrange(bottom[0].data.shape[0]):
            for j in xrange(bottom[0].data.shape[1]):
                cur_mask = mask[i,j,:,:].reshape((shape[2]*shape[3],))
                
                inds = np.argsort(cur_mask)
                inds = inds[::-1]
                weights[i,j,inds[0:min(self._num_sample,shape[2]*shape[3])]] = 1.0/min(self._num_sample,shape[2]*shape[3])
                if i == 0:
                    save_imagesc(mask[i,j] ,name= 'mask/mask_%d_%d'%(i,j))
                    #save_imagesc(weights[i,j,:].reshape((shape[2],shape[3])) ,name= 'mask/label_%d_%d'%(i,j))
                #tmp = np.where( mask[i,j,:,:] == mask[i,j,:,:].max())
                
                #weights[i,j,tmp[0][0],tmp[1][0]] = 1.0
        weights = weights.reshape((shape[0],shape[1],shape[2],shape[3]))        
        top[0].reshape(*(shape))
        top[0].data[...] = weights
        top[1].reshape(*(shape))
        top[1].data[...] = np.ones((shape[0],shape[1],shape[2],shape[3]))

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

