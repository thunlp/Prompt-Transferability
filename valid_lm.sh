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
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta_mlm/8.pkl \
    --pre_train_mlm True \
    --save_name mlm

exit



#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/REPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


#################################################
#################################################
##################Bert###########################
#################################################
#################################################



############
#Sentiment
############

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm




#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm




#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm



############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm

#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm



############
#RE
############
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/WikiREDPromptBert.config \
    --gpu $gpus \
    --checkpoint model/REPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm


CUDA_VISIBLE_DEVICES=$gpus python3 valid_lm.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptBert_mlm/15.pkl \
    --pre_train_mlm True \
    --save_name mlm




