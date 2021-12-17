gpus=0

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta

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
