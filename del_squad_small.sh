gpus=5


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/squadPromptT5.config \
    --gpu $gpus \
    --checkpoint model/squadPromptT5
'''


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/squadPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/squadPromptT5Small
