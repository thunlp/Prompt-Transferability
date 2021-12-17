gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptT5.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptT5


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRobertaLarge
'''
