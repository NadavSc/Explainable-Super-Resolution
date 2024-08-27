#!/bin/bash

python main.py --model 'srresbnn' \
               --model_type 'srresnet' \
               --model_path './modules/SRResBNN/models/SRResBNN/SRResBNN_4000.pt' \
               --mode 'test' \
               --num_workers 8 \
               --batch_size 1 \
               --bnn True \
               --cuda True
