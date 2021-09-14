gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastverbcornerPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/recastfactualityPromptRoberta
