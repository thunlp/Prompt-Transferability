mkdir RobertaForMaskedLM
gpus=7


################################
###########Roberta##############
################################

############
#Sentiment
############

'''
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed




#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed



#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \



#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \
'''



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \



#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \


############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \


############
#Other
############



#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \



CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True \



'''
################################
###########BERT#################
################################

############
#Sentiment
############

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptBert.config \
    --gpu $gpus \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


############
#Paraphrase
############


#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MRPCPromptBert.config \
    --gpu $gpus \



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QQPPromptBert.config \
    --gpu $gpus \



############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/RTEPromptBert.config \
    --gpu $gpus \



#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MNLIPromptBert.config \
    --gpu $gpus \



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WNLIPromptBert.config \
    --gpu $gpus \



############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WikiREDPromptBert.config \
    --gpu $gpus \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QNLIPromptBert.config \
    --gpu $gpus \



CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/STSBPromptBert.config \
    --gpu $gpus \
'''

