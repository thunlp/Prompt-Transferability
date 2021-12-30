gpus=7

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptRobertaSmall
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptT5Large.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptT5Large

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptRobertaLarge
'''
