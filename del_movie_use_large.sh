gpus=0

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesuseIMDBPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/movierationalesuseIMDBPromptRobertaLarge
