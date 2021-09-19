gpus=3

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta_label
