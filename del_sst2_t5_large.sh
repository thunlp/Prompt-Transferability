gpus=5

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRobertaSmall
'''


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptT5Large.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptT5Large


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRobertaLarge
'''
