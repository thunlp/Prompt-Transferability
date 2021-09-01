#CUDA_VISIBLE_DEVICES=$gpus

gpus=0



#################################################
#################################################
##################Roberta########################
#################################################
#################################################

############
#Sentiment
############

#restaurant
#Normal
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta



#MLM

#Task transfer, proj:False
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta \
    --replacing_prompt model/restaurantPromptRoberta




#Task transfer, proj:True
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta \
    --replacing_prompt model/restaurantPromptRoberta
    --task_transfer_projector


#Model transfer, proj:False
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert \
    --replacing_prompt model/restaurantPromptRoberta \
    --mode valid



#Model transfer, proj:True
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert \
    --replacing_prompt model/restaurantPromptRoberta \
    --model_transfer_projector


#restaurant
#Extract prompt
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta \
    --mode extract_prompt



#restaurant
#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta_mlm \
    --mode extract_prompt \
    --pre_train_mlm



#restaurant
#Eval mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta_mlm \
    --pre_train_mlm
