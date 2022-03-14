gpus=6


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptT5.config \
    --gpu $gpus
