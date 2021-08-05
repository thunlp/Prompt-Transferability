gpus=2

#cs_wiki
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/cs_wikiPromptRoberta_s1.config \
    --gpu $gpus \
    --pre_train_mlm True

#cs_wiki
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/cs_wikiPromptRoberta_s2.config \
    --gpu $gpus \
    --pre_train_mlm True
