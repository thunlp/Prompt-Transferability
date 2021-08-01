mkdir RobertaForMaskedLM
gpus=7


################################
###########Roberta##############
################################

############
#Sentiment
############

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed

exit

'''
#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRoberta.config \
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
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \


############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
'''



################################
###########BERT#################
################################

'''
############
#Sentiment
############

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptBert.config \
    --gpu $gpus \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
'''


############
#Paraphrase
############


#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptBert.config \
    --gpu $gpus \



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptBert.config \
    --gpu $gpus \



############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptBert.config \
    --gpu $gpus \



#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptBert.config \
    --gpu $gpus \



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WNLIPromptBert.config \
    --gpu $gpus \



############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WikiREDPromptBert.config \
    --gpu $gpus \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptBert.config \
    --gpu $gpus \



CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/STSBPromptBert.config \
    --gpu $gpus \

