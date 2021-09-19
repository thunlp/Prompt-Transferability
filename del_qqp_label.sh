gpus=1

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta_label
