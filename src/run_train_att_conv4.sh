#!/bin/bash
for i in $(seq 1.0 -0.2 0.0)
do 
    python ./src/train_att.py -c ./configs/mini/softmax/conv4_att.config --gpu 1  --loss-alpha 1.0 --loss-beta $i --log-info --top-k 16
done