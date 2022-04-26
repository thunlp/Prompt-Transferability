
# Bert: BertMedium, Bert (BertBase)
# Roberta: Roberta, RobertaBase (BertBase)
# T5: T5Small, T5 (T5Base), T5Large
gpus=7
DATASET="SST2"
BACKBONE="Roberta"


CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/{DATASET}Prompt${BACKBONE}.config \
    --gpu $gpus \
    --checkpoint model/${DATASET}Prompt${BACKBONE} \




'''
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
    --checkpoint model/restaurantPromptRoberta \


#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta  \


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta \



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta \

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta \


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta \

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta \



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta \



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/REPromptRoberta \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta \

CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta \


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
    --checkpoint model/restaurantPromptBert \




#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert \




#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert \



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert \

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert \


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptBert \

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptBert \



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptBert \



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WikiREDPromptBert.config \
    --gpu $gpus \
    --checkpoint model/REPromptBert \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptBert \

CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptBert \
'''




