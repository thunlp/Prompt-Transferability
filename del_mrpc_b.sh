gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert
