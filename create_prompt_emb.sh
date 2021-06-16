#rm -rf task_prompt_emb/*

gpus=2
#CUDA_VISIBLE_DEVICES=$gpus

'''
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

###

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/MNLIPromptRoberta/1.pkl \


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/WNLIPromptRoberta/1.pkl \


#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/MRPCPromptRoberta/1.pkl \


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/QNLIPromptRoberta/1.pkl \



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/QQPPromptRoberta/1.pkl \
'''

#STSB
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/STSBPromptRoberta/1.pkl \
