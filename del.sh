mkdir RobertaForMaskedLM
gpus=1

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
