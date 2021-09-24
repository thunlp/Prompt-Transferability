gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptBert.config \
    --gpu $gpus \
    --checkpoint model/snliPromptBert
