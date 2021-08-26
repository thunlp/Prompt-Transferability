gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/anliPromptRoberta.config \
    #--checkpoint model/anliPromptRoberta/15.pkl \
    --gpu $gpus
