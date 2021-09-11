gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert
