gpus=7

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptBert.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptBert
