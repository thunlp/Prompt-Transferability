gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRoberta
