gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptBert.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptBert
