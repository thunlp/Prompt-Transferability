#CUDA_VISIBLE_DEVICES=$gpus

gpus=6

############
#Sentiment
############

'''
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta/15.pkl \
    --model_prompt Bert-base


#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta/15.pkl \


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta/15.pkl \


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta/15.pkl \



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta/15.pkl \

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta/15.pkl \


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta/15.pkl \

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta/15.pkl \



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta/15.pkl \



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt/15.pkl \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta/15.pkl \


CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta/15.pkl \


'''

################################
###########BERT#################
################################


############
#Sentiment
############

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert/15.pkl \
    --model_prompt Roberta-base

exit


#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert/15.pkl \


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert/15.pkl \




#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert/15.pkl \



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert/15.pkl \

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert/15.pkl \


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptBert/15.pkl \

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptBert/15.pkl \



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptBert/15.pkl \



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt/15.pkl \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptBert/15.pkl \


CUDA_VISIBLE_DEVICES=$gpus python3 valid_roberta-bert_prompt.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptBert/15.pkl \



