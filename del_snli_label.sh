gpus=0

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/snliPromptRoberta_label
