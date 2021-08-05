gpus=5

#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta_s1.config \
    --gpu $gpus \
    --pre_train_mlm True

#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta_s2.config \
    --gpu $gpus \
    --pre_train_mlm True
