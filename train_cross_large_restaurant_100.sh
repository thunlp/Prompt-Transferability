#mkdir RobertaForMaskedLM
gpus=3

############
#Sentiment
############

#model_prompt="bert-base"
model_prompt="roberta-base"

#restaurant
#CUDA_VISIBLE_DEVICES=$gpus python3 train_projector.py --config config/projectorPromptRoberta.config \
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPromptRobertaLarge_restaurant_100.config \
    --gpu $gpus \
    --model_prompt $model_prompt

