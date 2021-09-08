#mkdir RobertaForMaskedLM
gpus=1

############
#Sentiment
############

model_prompt="bert-medium"

# model --> bert-base
# bert-medium promt --> projector() --> for bert-base prompt

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/cross_mlmPromptRobertaMedium.config \
    --gpu $gpus \
    --model_prompt $model_prompt \
    --pre_train_mlm True

