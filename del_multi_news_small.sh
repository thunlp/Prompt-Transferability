gpus=6


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/multi_newsPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/multi_newsPromptT5Small
