gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptRoberta_label
