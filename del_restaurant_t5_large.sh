gpus=6


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta
'''

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptT5Large.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptT5Large

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRobertaLarge
'''

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRobertaSmall.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRobertaSmall
'''
