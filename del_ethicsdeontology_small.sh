gpus=2

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptRoberta
'''


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/ethicsdeontologyPromptT5Small
