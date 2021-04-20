#!/bin/bash
# for i in $(seq 1.0 -0.2 0.8)
# do 
#     python ./src/train_att.py -c ./configs/mini/softmax/conv4_att.config --gpu 2  --loss-alpha 1.0 --loss-beta $i --log-info --top-k 16
# done


python ./src/train.py -c ./configs/mini/softmax/conv4.config --gpu 0 --log-info
python ./src/train.py -c ./configs/mini/softmax/resnet18.config --gpu 0 --log-info
python ./src/train.py -c ./configs/mini/softmax/resnet10.config --gpu 0 --log-info