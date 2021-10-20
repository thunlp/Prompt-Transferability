gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2uselaptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2uselaptopPromptRoberta
    #--checkpoint model/SST2useIMDBPromptRoberta
