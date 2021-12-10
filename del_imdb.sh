gpus=0

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptT5.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptT5
'''


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRobertaLarge
