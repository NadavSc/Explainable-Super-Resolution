#!/bin/bash

python main.py --model 'srresebnn' \
               --model_type 'srresnet' \
               --mode 'train' \
               --num_workers 8 \
               --batch_size 128 \
               --bnn True \
               --cuda True \
               --fine_tuning False \
               --pre_train_epoch 4000 \
               --fine_train_epoch 0 \
               --fine_tuning False

