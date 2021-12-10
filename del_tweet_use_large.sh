gpus=2

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetuserestaurantPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/tweetuserestaurantPromptRobertaLarge
