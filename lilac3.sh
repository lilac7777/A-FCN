#!/bin/bash
FIL="lilac3.txt"
mynum=7000
while true
do



nvidia-smi -i 2 > $FIL
Usedstr=`grep "250W" ${FIL}`
Used=`grep "250W" ${FIL} | awk '{print $8}'`
substring=MiB
Used=${Used%$substring}

if [ $Used -lt $mynum ]; then  
    XAUTHORITY=${HOME}/.Xauthority ./experiments/scripts/softmax_hard_rfcn.sh 3 ResNet-50 pascal_voc 
fi  





sleep 1
done