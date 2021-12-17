gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptT5.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptT5
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRobertaLarge
'''
