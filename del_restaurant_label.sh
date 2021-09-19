gpus=0

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta_label
