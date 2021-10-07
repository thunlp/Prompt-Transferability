gpus=0

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastneruseMNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/recastneruseMNLIPromptRoberta
