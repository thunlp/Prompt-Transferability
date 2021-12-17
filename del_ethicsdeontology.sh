gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptRoberta


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
