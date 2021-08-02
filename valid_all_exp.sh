#CUDA_VISIBLE_DEVICES=$gpus

gpus=5



#################################################
#################################################
##################Roberta########################
#################################################
#################################################

############
#Sentiment
############

'''
#restaurant
#Normal
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta/15.pkl \



#MLM

#Task transfer, proj:False
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta/15.pkl \
    --replacing_prompt restaurantPromptRoberta




#Task transfer, proj:True
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta/15.pkl \
    --replacing_prompt restaurantPromptRoberta \
    --task_transfer_projector


#Model transfer, proj:False
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert/15.pkl \
    --replacing_prompt restaurantPromptRoberta

'''


#Model transfer, proj:True
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert/15.pkl \
    --replacing_prompt restaurantPromptRoberta \
    --model_transfer_projector





