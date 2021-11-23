gpus=2

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta
'''


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptT5.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptT5
