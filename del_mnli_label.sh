gpus=7

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta_label
