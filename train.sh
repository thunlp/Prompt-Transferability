mkdir RobertaForMaskedLM


#SST-2
'''
python3 train.py --config config/SST2PromptRoberta.config \
    --gpu 0,1,2,3 \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
'''


#RET
#Remove prompts between two sentences
'''
python3 train.py --config config/RTEPromptRoberta.config \
    --gpu 5 \
'''


#RE
python3 train.py --config config/REPrompt.config \
    --gpu 5 \
