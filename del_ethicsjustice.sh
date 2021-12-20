gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticePromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/ethicsjusticePromptRobertaSmall

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticePromptT5.config \
    --gpu $gpus \
    --checkpoint model/ethicsjusticePromptT5
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticePromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/ethicsjusticePromptRobertaLarge
'''
