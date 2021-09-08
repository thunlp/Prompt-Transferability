mkdir RobertaForMaskedLM
gpus=7

############
#Sentiment
############

#model_prompt="bert-medium"
model_prompt="bert-medium"
#model_prompt="roberta-base"

#restaurant
#CUDA_VISIBLE_DEVICES=$gpus python3 train_projector.py --config config/projectorPromptRoberta.config \
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPromptBertMedium.config \
    --gpu $gpus \
    --model_prompt $model_prompt

