#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_alt_opt.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is only pascal_voc for now
#
# Example:
# ./experiments/scripts/faster_rcnn_alt_opt.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=ResNet-50
NET_lc=${NET,,}
DATASET=pascal_voc

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_val"
    PT_DIR="coco"
    ITERS=40000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

time ./tools/train_rfcn_fix_hard.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights /net/wujial/py-R-FCN/data/rfcn_models/resnet50_rfcn_final.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --imdb_test ${TEST_IMDB} \
  --cfg experiments/cfgs/rfcn_alt_opt_5step_ohem.yml \
  --model 'sigmoid_agnostic_mask'
  ${EXTRA_ARGS}

set +x
