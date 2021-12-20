gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/snliPromptRobertaSmall

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptT5.config \
    --gpu $gpus \
    --checkpoint model/snliPromptT5
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/snliPromptRobertaLarge
'''
