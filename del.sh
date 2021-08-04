mkdir RobertaForMaskedLM
gpus=6


################################
###########Roberta##############
################################

############
#Sentiment
############

'''
#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptRoberta_s1.config \
    --gpu $gpus \
    --pre_train_mlm True
'''


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptRoberta_s2.config \
    --gpu $gpus \
    --pre_train_mlm True



'''
#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta_s1.config \
    --gpu $gpus \
    --pre_train_mlm True

#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta_s2.config \
    --gpu $gpus \
    --pre_train_mlm True
'''
