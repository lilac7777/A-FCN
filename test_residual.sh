#!/bin/bash
FIL="test_res.txt"

./experiments/scripts/test_rfcn.sh 3 /net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/residual_pos_neg_attention_1/resnet50_rfcn_mask_ohem_iter_60000.caffemodel residual_pos_neg_attention_1 >> $FIL
./experiments/scripts/test_rfcn.sh 3 /net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/residual_pos_neg_attention_2/resnet50_rfcn_mask_ohem_iter_45000.caffemodel residual_pos_neg_attention_2 >> $FIL
./experiments/scripts/test_rfcn.sh 3 /net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/residual_pos_neg_attention_4/resnet50_rfcn_mask_ohem_iter_60000.caffemodel residual_pos_neg_attention_4 >> $FIL
./experiments/scripts/test_rfcn.sh 3 /net/wujial/py-R-FCN/models/pascal_voc/ResNet-50/residual_pos_neg_attention_7/resnet50_rfcn_mask_ohem_iter_60000.caffemodel residual_pos_neg_attention_7 >> $FIL
