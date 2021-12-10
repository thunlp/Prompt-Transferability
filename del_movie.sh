gpus=2

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptRoberta
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptT5.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptT5
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/movierationalesPromptRobertaLarge
