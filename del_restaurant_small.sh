gpus=6

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptT5Small
