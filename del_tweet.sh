gpus=3

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRobertaSmall

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptT5.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptT5
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRobertaLarge
'''
