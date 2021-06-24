mkdir RobertaForMaskedLM
gpus=4

############
#Sentiment
############

############
#Other
############


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \


