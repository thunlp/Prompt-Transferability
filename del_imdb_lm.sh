gpus=6

#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta_mlm_s1/15.pkl \
    --pre_train_mlm True

#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta_mlm_s2/15.pkl \
    --pre_train_mlm True


#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert_mlm_s1/15.pkl \
    --pre_train_mlm True

#imdb
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert_mlm_s2/15.pkl \
    --pre_train_mlm True
