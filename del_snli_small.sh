gpus=0

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/snliPromptRoberta
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/snliPromptT5Small
