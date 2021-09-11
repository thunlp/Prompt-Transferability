gpus=7

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert
