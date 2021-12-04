#mkdir RobertaForMaskedLM
gpus=6

############
#Sentiment
############

#model_prompt="Bert-base"
model_prompt="T5-base"
#model_prompt="Roberta-base"

#restaurant
#CUDA_VISIBLE_DEVICES=$gpus python3 train_projector.py --config config/projectorPromptRoberta.config \
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPromptRoberta_laptop_100_t5_to_roberta.config \
    --gpu $gpus \
    --model_prompt $model_prompt
    #--checkpoint roberta-base \
    #--seed

