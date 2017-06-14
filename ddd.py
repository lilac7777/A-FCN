from plot_fig import *
import scipy.io as sio
netdata = sio.loadmat("/net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/relu_hard_rfcn_god/resnet50_rfcn_ohem_iter_10000.mat")
mask = netdata['Sig_mask']
for i in range(25):
    vis_detections(mask[0,i,:,:])
