gpus=4

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRobertaSmall
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptT5Large.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptT5Large


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRobertaLarge
'''
