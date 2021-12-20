gpus=3


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptT5.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptT5
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRobertaLarge
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRobertaSmall
