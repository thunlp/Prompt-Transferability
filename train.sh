mkdir RobertaForMaskedLM
gpus=1


'''
#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
'''

'''
#RTE
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
'''


'''
#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/REPrompt.config \
    --gpu $gpus \
'''


#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \

'''
#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \


#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
'''


