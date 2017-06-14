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

class WeakWeightLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        #self._num_classes = layer_params['num_classes']
        self._num_hard_mask = layer_params['num_hard_mask_sqrt']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(1, 21,self._num_hard_mask, self._num_hard_mask)
        self._name_to_top_map['hard_mask'] = idx
        idx += 1
        
        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        bottom_shape = bottom[0].data.shape
        mask = np.zeros(bottom_shape[0],self._num_hard_mask*self._num_hard_mask,self._num_hard_mask,self._num_hard_mask)
        mask_shape = mask.shape
        for i in xrange(self._num_hard_mask):
            for j in xrange(self._num_hard_mask):
                mask[:,i*self._num_hard_mask + j,i,j] = 1
        top[0].reshape(*(mask_shape))
        top[0].data[...] = mask
        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

