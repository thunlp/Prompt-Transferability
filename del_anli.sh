gpus=0


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/anliPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/anliPromptRoberta
'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/anliPromptT5.config \
    --gpu $gpus \
    --checkpoint model/anliPromptT5
