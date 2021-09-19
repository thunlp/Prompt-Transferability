gpus=2

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta_label
