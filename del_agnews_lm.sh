gpus=4

#agnews
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/agnewsPromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptRoberta_mlm_s1/15.pkl \
    --pre_train_mlm True



#agnews
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/agnewsPromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptRoberta_mlm_s2/15.pkl \
    --pre_train_mlm True

#agnews
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/agnewsPromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptBert_mlm_s1/15.pkl \
    --pre_train_mlm True




#agnews
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/agnewsPromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptBert_mlm_s2/15.pkl \
    --pre_train_mlm True
