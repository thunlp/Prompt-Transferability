gpus=3

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptRoberta_label
