gpus=0

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta
'''


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptT5Small
