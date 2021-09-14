gpus=2

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastmegaveridicalityPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/recastmegaveridicalityPromptRoberta
