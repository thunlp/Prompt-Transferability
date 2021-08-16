mkdir RobertaForMaskedLM
gpus=7

############
#Sentiment
############

model_prompt="bert-base"


CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/cross_mlmPromptRoberta.config \
    --gpu $gpus \
    --model_prompt $model_prompt
exit

'''
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPromptRoberta.config \
    --gpu $gpus \
    --model_prompt $model_prompt
'''

