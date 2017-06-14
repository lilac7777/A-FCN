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
class SoftPlusLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    
    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        bottom_shape = bottom[0].shape
        idx = 0
        top[idx].reshape(bottom_shape[0],bottom_shape[1],bottom_shape[2],bottom_shape[3])
        

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        bottom_shape = bottom[0].shape
        top[0].reshape(bottom_shape[0],bottom_shape[1],bottom_shape[2],bottom_shape[3])
        top[0].data[...] = np.log( 1.0 + np.exp(bottom[0].data[...]))
        
        
        

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        top_shape = top[0].shape
        bottom[0].diff[...] = top[0].diff[...]*(np.exp(bottom[0].data[...])/(1.0 + np.exp(bottom[0].data[...]) ))

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

