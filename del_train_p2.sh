mkdir RobertaForMaskedLM
gpus=1



############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \



############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
