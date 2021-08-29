gpus=7

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptBert.config \
    --gpu $gpus

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptRoberta.config \
    --gpu $gpus
