gpus=2

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta_label
