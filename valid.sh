#SST
'''
python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu 0,1,2,3 \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta/1.pkl \
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
'''

#RTE
'''
python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu 0 \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/RTE_BERT/1.pkl \
'''


#RE
python3 valid.py --config config/REPrompt.config \
    --gpu 0 \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/REPrompt/15.pkl \
