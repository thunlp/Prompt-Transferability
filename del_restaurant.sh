gpus=3

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta
