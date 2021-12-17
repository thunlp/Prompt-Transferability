gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticePromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/ethicsjusticePromptRoberta

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
