gpus=7

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptT5.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptT5
'''


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRobertaLarge
