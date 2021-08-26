gpus=4


############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True


############
#NLI
############

#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True



#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True


############
#RE
############

#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WikiREDPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True


############
#Other
############



#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True



CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True


