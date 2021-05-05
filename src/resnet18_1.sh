#!/bin/bash
# for i in $(seq 0.1 0.2 1.0)
# do 
#     python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2  --loss-alpha 1.0 --loss-beta $i --log-info --pool-type max_pool --top-k 32
# done


    # python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2  --loss-alpha 1.0 --loss-beta 1.0 --log-info --pool-type max_pool --top-k 32
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 0  --loss-alpha 1.0 --loss-beta 0.2 --log-info --pool-type max_pool --top-k 32 --batch-size 100
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 1  --loss-alpha 1.0 --loss-beta 0.5 --log-info --pool-type max_pool --top-k 32 --batch-size 100
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2  --loss-alpha 1.0 --loss-beta 0.8 --log-info --pool-type max_pool --top-k 32 --batch-size 100