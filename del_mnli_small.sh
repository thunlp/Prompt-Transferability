gpus=4

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptT5Small
