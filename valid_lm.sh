#CUDA_VISIBLE_DEVICES=$gpus

gpus=6



#################################################
#################################################
##################Roberta########################
#################################################
#################################################

############
#Sentiment
############
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta_mlm \
    --pre_train_mlm



#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta_mlm \
    --pre_train_mlm




#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta_mlm \
    --pre_train_mlm



#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta_mlm \
    --pre_train_mlm


exit

'''
############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta_mlm \
    --pre_train_mlm

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta_mlm \
    --pre_train_mlm


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta_mlm \
    --pre_train_mlm

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta_mlm \
    --pre_train_mlm



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta_mlm \
    --pre_train_mlm



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/REPromptRoberta_mlm \
    --pre_train_mlm


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta_mlm \
    --pre_train_mlm


CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta_mlm \
    --pre_train_mlm


#################################################
#################################################
##################Bert###########################
#################################################
#################################################



############
#Sentiment
############

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert_mlm \
    --pre_train_mlm




#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert_mlm \
    --pre_train_mlm




#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert_mlm \
    --pre_train_mlm


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert_mlm \
    --pre_train_mlm



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert_mlm \
    --pre_train_mlm

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert_mlm \
    --pre_train_mlm


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptBert_mlm \
    --pre_train_mlm

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptBert_mlm \
    --pre_train_mlm



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptBert_mlm \
    --pre_train_mlm



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WikiREDPromptBert.config \
    --gpu $gpus \
    --checkpoint model/REPromptBert_mlm \
    --pre_train_mlm


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptBert_mlm \
    --pre_train_mlm


CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptBert_mlm \
    --pre_train_mlm
'''




