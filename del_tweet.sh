gpus=1

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptRoberta
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptT5.config \
    --gpu $gpus \
    --checkpoint model/tweetevalsentimentPromptT5
