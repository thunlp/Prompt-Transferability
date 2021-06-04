mkdir RobertaForMaskedLM
gpus=5


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#RET
#Remove prompts between two sentences
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \


#RE
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/REPrompt.config \
    --gpu $gpus \
