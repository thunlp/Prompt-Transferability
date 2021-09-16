gpus=1

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert
