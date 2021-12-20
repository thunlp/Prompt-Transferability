gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRobertaSmall

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptT5.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptT5
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRobertaLarge
'''
