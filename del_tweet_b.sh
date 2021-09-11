gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptBert.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptBert
