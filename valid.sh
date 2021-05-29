#SST
'''
python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu 0,1,2,3 \
    --checkpoint  /mnt/datadisk0/suyusheng/prompt/prompt/model/SST2PromptRoberta/1.pkl \
    #--result /mnt/datadisk0/suyusheng/prompt/prompt/model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
'''

#RTE
python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu 0 \
    --checkpoint  /mnt/datadisk0/suyusheng/prompt/prompt/model/RTE_BERT/1.pkl \
