gpus=1

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/emobankarousalPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/emobankarousalPromptRoberta
