gpus=3

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptT5Small
