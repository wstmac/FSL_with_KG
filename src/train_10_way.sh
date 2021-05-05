#!/bin/bash
python ./src/train_att.py -c ./configs/mini/softmax/conv4_att.config --gpu 1 --meta-val-way 10 --log-info --loss-alpha 1.0 --loss-beta 0.5
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 1 --meta-val-way 10 --log-info --loss-alpha 1.0 --loss-beta 0.5
# python ./src/train.py -c ./configs/mini/softmax/resnet10.config --gpu 1 --meta-val-way 10 --log-info