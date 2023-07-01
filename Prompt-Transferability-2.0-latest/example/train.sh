#!/bin/bash

python example/train.py \
        --output_dir outputs \
        --dataset sst2 \
        --learning_rate 1e-2 \
        --num_train_epochs 3 \
        --save_total_limit 1 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end true \
        --metric_for_best_model combined_score
