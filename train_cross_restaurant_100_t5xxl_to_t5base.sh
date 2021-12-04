#mkdir RobertaForMaskedLM
gpus=2

############
#Sentiment
############

model_prompt="t5-base"

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPromptT5_restaurant_100_t5xxl_to_t5base.config \
    --gpu $gpus \
    --model_prompt $model_prompt
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed

