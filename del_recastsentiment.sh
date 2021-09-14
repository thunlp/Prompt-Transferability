gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastsentimentPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/recastsentimentPromptRoberta
