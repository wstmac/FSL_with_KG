#!/bin/bash
# for i in $(seq 0.1 0.2 1.0)
# do 
#     python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2  --loss-alpha 1.0 --loss-beta $i --log-info --pool-type max_pool --top-k 32
# done


    # python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2  --loss-alpha 1.0 --loss-beta 1.0 --log-info --pool-type max_pool --top-k 32
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 0  --loss-alpha 1.0 --loss-beta 0.2 --log-info --pool-type max_pool --top-k 32 --batch-size 100
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 1  --loss-alpha 1.0 --loss-beta 0.5 --log-info --pool-type max_pool --top-k 32 --batch-size 100
python ./src/train_att.py -c ./configs/mini/softmax/resnet18_att.config --gpu 2  --loss-alpha 1.0 --loss-beta 0.8 --log-info --pool-type max_pool --top-k 32 --batch-size 100


python ./src/train.py -c ./configs/mini/softmax/resnet50.config --gpu 1
python ./src/train_att.py -c ./configs/mini/softmax/resnet50_att.config --gpu 2  --loss-alpha 1.0 --loss-beta 0.8 --log-info --pool-type max_pool --top-k 32 --batch-size 12 


CUDA_VISIBLE_DEVICES=1,2,3 python ./src/ddp_train_att.py -c ./configs/mini/softmax/resnet50_att.config --world-size 1 --rank 0 --batch-size 96 --moco-k 1920 --moco-t 0.1 --epoch 800 --loss-alpha 0.2 --loss-beta 1.

python ./src/evaluate.py -c ./configs/mini/softmax/resnet18_att.config --model-dir 2021-05-23_11-46-04 --model-name checkpoint_699.pth.tar --gpu 2 --log-info --pool-type avg_pool --top-k 32