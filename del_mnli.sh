gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRobertaSmall

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptT5.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptT5
'''


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRobertaLarge
'''
