gpus=7

#scierc
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/sciercPromptRoberta_s1.config \
    --gpu $gpus \
    --pre_train_mlm True

#scierc
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/sciercPromptRoberta_s2.config \
    --gpu $gpus \
    --pre_train_mlm True


#scierc
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/sciercPromptBert_s1.config \
    --gpu $gpus \
    --pre_train_mlm True

#scierc
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/sciercPromptBert_s2.config \
    --gpu $gpus \
    --pre_train_mlm True











#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta_s1.config \
    --gpu $gpus \
    --pre_train_mlm True

#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta_s2.config \
    --gpu $gpus \
    --pre_train_mlm True


#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptBert_s1.config \
    --gpu $gpus \
    --pre_train_mlm True

#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptBert_s2.config \
    --gpu $gpus \
    --pre_train_mlm True
