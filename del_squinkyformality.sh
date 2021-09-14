gpus=7

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/squinkyformalityPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/squinkyformalityPromptRoberta
