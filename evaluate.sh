#!/bin/bash

python main.py --model 'srresebnn' \
               --model_type 'srresnet' \
               --db_valid_sr_path './modules/SRResBNN/results/SRResBNN' \
               --mode 'evaluate' \
               --bnn True \
               --cuda True \


