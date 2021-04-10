#!/bin/bash
for i in $(seq 1.0 -0.2 0.0)
do 
    python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2  --loss-alpha 1.0 --loss-beta $i --log-info --pool-type max_pool
done