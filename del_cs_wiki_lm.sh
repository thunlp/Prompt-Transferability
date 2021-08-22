gpus=5

#cs_wiki
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/cs_wikiPromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptRoberta_mlm_s1/15.pkl \
    --pre_train_mlm True

#cs_wiki
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/cs_wikiPromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptRoberta_mlm_s2/15.pkl \
    --pre_train_mlm True


#cs_wiki
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/cs_wikiPromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptBert_mlm_s1/15.pkl \
    --pre_train_mlm True

#cs_wiki
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/cs_wikiPromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptBert_mlm_s2/15.pkl \
    --pre_train_mlm True
