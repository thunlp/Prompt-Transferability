mkdir RobertaForMaskedLM
gpus=2

############
#Sentiment
############


#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train_projector.py --config config/projectorPromptRoberta.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed

