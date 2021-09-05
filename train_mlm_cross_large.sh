mkdir RobertaForMaskedLM
gpus=1

############
#Sentiment
############

model_prompt="bert-base"

# model --> roberta
# promt --> projector(bert prompt) --> for Roberta prompt

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/cross_mlmPromptRobertaLarge.config \
    --gpu $gpus \
    --model_prompt $model_prompt \
    --pre_train_mlm True

exit

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPromptRoberta.config \
    --gpu $gpus \
    --model_prompt $model_prompt
'''

