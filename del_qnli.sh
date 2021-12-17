gpus=6

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptT5.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptT5
'''


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRobertaLarge
'''
