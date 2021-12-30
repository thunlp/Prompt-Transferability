gpus=5

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRobertaSmall
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptT5Large.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptT5Large

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRobertaLarge
'''
