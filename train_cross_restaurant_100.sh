#mkdir RobertaForMaskedLM
gpus=7

############
#Sentiment
############

model_prompt="bert-base"

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPromptRoberta_restaurant_100.config \
    --gpu $gpus \
    --model_prompt $model_prompt
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed

