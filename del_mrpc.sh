gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta
