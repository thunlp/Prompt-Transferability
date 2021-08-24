gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/recastfactualityPromptBert.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptBert.config \
    --gpu $gpus
