gpus=2

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta_label
