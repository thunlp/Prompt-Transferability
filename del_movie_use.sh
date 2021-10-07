gpus=7

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesuseIMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/movierationalesuseIMDBPromptRoberta
