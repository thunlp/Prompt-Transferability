gpus=5

'''
#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptBert.config \
    --gpu $gpus
'''


#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRoberta.config \
    --gpu $gpus


