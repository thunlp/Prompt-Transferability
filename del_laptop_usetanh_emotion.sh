gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptRobertausetanhemotionPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptop_base_emotionPromptRoberta_tanh
