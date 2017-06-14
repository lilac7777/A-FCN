import sys, os
sys.path.insert(0, '/net/wujial/py-R-FCN/tools')
import _init_paths
from fast_rcnn.train import get_training_roidb, train_net,SolverWrapper, filter_roidb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals, imdb_rpn_compute_stats
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil
import caffe
from train_rfcn_alt_opt_5stage import get_roidb, _init_caffe
import scipy.io as sio


output_cache='stage1_rfcn_final.caffemodel'
rpn_file='/net/wujial/py-R-FCN/output/rfcn_alt_opt_5step_ohem/voc_2007_trainval/stage1_rpn_final_proposals.pkl'
#init_model = '/net/wujial/py-R-FCN/data/imagenet_models/VGG_ILSVRC_16_layers.caffemodel'
init_model = '/net/wujial/py-R-FCN/data/imagenet_models/ResNet-50-model.caffemodel'
max_iters = 10000
#solver = '/net/wujial/py-R-FCN/models/pascal_voc/vgg16/soft_rfcn_alt_opt_5step_ohem/stage1_rfcn_ohem_solver80k120k.pt'
imdb_name = 'voc_2007_trainval'
solver = '/net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/soft_rfcn_alt_opt_5step_ohem/stage1_rfcn_ohem_solver80k120k.pt'




NET_FINAL=`grep "Final model" ${LOG} | awk '{print $3}'`
RPN_FINAL=`grep "Final RPN" ${LOG} | awk '{print $3}'`
set -x
time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/rfcn_alt_opt_5step_ohem/rfcn_test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --rpn_file ${RPN_FINAL} \
  --cfg experiments/cfgs/rfcn_alt_opt_5step_ohem.yml \
  --num_dets 400
  ${EXTRA_ARGS}










print 'Init model: {}'.format(init_model)
print 'RPN proposals: {}'.format(rpn_file)
print('Using config:')
pprint.pprint(cfg)
_init_caffe(cfg)
roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file)
output_dir = get_output_dir(imdb)
print 'Output will be saved to `{:s}`'.format(output_dir)
# Train R-FCN
# Send R-FCN model path over the multiprocessing queue
final_caffemodel = os.path.join(output_dir, output_cache)
roidb = filter_roidb(roidb)
sw = SolverWrapper(solver, roidb, output_dir, init_model)
sw.solver.step(1)
net = sw.solver.net
netdata = dict()
saveto = 'test.mat'
netdata['data'] = net.blobs['data'].data
netdata['res5c'] = net.blobs['res5c'].data

sio.savemat(saveto,netdata)


if os.path.exists(final_caffemodel):
    print 'has done'
else:
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rfcn_model_path = model_paths[-1]
     # Send final model path through the multiprocessing queue
    shutil.copyfile(rfcn_model_path, final_caffemodel)
    #queue.put({'model_path': final_caffemodel})
    
    
    