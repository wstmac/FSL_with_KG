#!/bin/bash
python ./src/train.py -c ./configs/mini/softmax/conv4.config --gpu 1 --meta-val-way 10 --log-info
python ./src/train.py -c ./configs/mini/softmax/resnet18.config --gpu 1 --meta-val-way 10 --log-info
python ./src/train.py -c ./configs/mini/softmax/resnet10.config --gpu 1 --meta-val-way 10 --log-info