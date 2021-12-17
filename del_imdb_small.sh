gpus=7



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptT5Small
