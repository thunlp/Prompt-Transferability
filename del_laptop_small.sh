gpus=3

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptT5Small
