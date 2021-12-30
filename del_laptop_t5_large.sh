gpus=2

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRobertaSmall
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptT5Large.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptT5Large


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRobertaLarge
'''
