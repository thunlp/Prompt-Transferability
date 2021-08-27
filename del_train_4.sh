gpus=6



############
#Paraphrase
############

'''
#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True
'''


'''
#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True
'''




############
#NLI
############
'''
#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True
'''


#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True



#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True





############
#RE
############
'''
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/WikiREDPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True
'''


############
#Other
############


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True





CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True
