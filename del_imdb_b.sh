gpus=1

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert
