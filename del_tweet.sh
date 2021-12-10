gpus=7

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRoberta
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptT5.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptT5
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRobertaLarge
