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

class ConfWeightLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        #self._num_classes = layer_params['num_classes']
        #self._num_sample = layer_params['num_sample']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(1, cfg.TRAIN.mask_num ,1,1)
        self._name_to_top_map['conf_label'] = idx
        idx += 1
        
        
        
        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        labels = bottom[1].data #shape n,M,H,W
        data = bottom[0].data #shape n,M,K
        shape = data.shape
        data = data.reshape((shape[0], 21 ,cfg.TRAIN.mask_num))
        datalabel = np.argmax(data, axis = 1)#shape n,K,
        datalabel = datalabel.reshape((shape[0],cfg.TRAIN.mask_num))
        labels = labels[:,:,0,0]
        #print labels.shape
        labels = labels.reshape((shape[0],21))
        labels = np.argmax(labels, axis = 1)#shape n,H,W
        labels = labels.reshape((shape[0],1))#shape n,1
        conf_label = np.zeros((shape[0],cfg.TRAIN.mask_num ))
        for i in xrange(cfg.TRAIN.mask_num):
            tmp = datalabel[:,i].reshape((shape[0],1))
            conf_label[:,i] = (tmp == labels).reshape((shape[0])).astype('float32')
        conf_label = conf_label.reshape((shape[0],cfg.TRAIN.mask_num,1,1))
        shape = conf_label.shape
        top[0].reshape(*(shape))
        top[0].data[...] = conf_label
        
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

