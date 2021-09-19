gpus=3

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastnerPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/recastnerPromptRoberta_label
