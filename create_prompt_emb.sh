gpus=5
#CUDA_VISIBLE_DEVICES=$gpus

#SST
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta/1.pkl \
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed



#RTE
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/RTE_BERT/1.pkl \


#RE
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/REPrompt/15.pkl \
