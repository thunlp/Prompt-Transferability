#CUDA_VISIBLE_DEVICES=$gpus

gpus=7

#MODEL_PROMPT="Roberta-base"
#MODEL_PROMPT="Bert-base"
#MODEL_PROMPT="Random"
PROJECTOR="model/projectPromptRoberta"

############
#Sentiment
############
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta \
    --replacing_prompt task_prompt_emb/restaurantPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR





#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta \
    --replacing_prompt task_prompt_emb/laptopPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR



#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta \
    --replacing_prompt task_prompt_emb/IMDBPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR



#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta \
    --replacing_prompt task_prompt_emb/SST2PromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta \
    --replacing_prompt task_prompt_emb/MRPCPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR


#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta \
    --replacing_prompt task_prompt_emb/QQPPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta \
    --replacing_prompt task_prompt_emb/RTEPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR


#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta \
    --replacing_prompt task_prompt_emb/MNLIPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta \
    --replacing_prompt task_prompt_emb/WNLIPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR



############
#RE
############
#RE
'''
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR
'''


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta \
    --replacing_prompt task_prompt_emb/QNLIPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR


CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta \
    --replacing_prompt task_prompt_emb/STSBPromptBert \
    --model_transfer_projector \
    --projector $PROJECTOR



################################
###########BERT#################
################################
'''

############
#Sentiment
############
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR



#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR




#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR




#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR




############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR


#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR


#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR




#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR




############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR



CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptBert \
    --replacing_prompt  \
    --model_transfer_projector \
    --projector $PROJECTOR
'''




