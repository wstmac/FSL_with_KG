#!/bin/bash

python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_99.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32
python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_199.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32
python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_299.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32
python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_399.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32
python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_499.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32
python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_599.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32
python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_799.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32

python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_12-12-22 --model-name checkpoint_99.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32
python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_12-12-22 --model-name checkpoint_199.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32