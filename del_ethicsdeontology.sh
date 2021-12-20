gpus=3

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptRobertaSmall


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptT5.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptT5
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptRobertaLarge
'''
