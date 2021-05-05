#!/bin/bash
python ./src/train_att.py -c ./configs/mini/softmax/conv4_att.config --gpu 2 --meta-val-way 20 --log-info --loss-alpha 1.0 --loss-beta 0.5
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2 --meta-val-way 20 --log-info --loss-alpha 1.0 --loss-beta 0.5
# python ./src/train.py -c ./configs/mini/softmax/resnet10.config --gpu 2 --meta-val-way 20 --log-info


python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2 --meta-val-way 16 --debug --loss-alpha 1.0 --loss-beta 0.5 --model-dir 2021-04-21_07-02-41