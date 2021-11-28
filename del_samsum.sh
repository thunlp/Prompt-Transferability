gpus=3


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/samsumPromptT5.config \
    --gpu $gpus \
    --checkpoint model/samsumPromptT5
