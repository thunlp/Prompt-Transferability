gpus=7


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/samsumPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/samsumPromptT5Small
