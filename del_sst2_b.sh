gpus=1

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert
