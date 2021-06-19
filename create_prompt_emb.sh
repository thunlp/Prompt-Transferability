#rm -rf task_prompt_emb/*

echo Do you wanna rewrite task emb in the task_prompt_emb y/n ?
read ans

if [ $ans = "y" ]
then
    rm -rf task_prompt_emb/*
else
    echo "Do not rewrite"
    #exit
fi


gpus=2
#CUDA_VISIBLE_DEVICES=$gpus


#SST
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta/15.pkl \
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed



'''
#RTE
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/RTE_BERT/3.pkl \


#RE
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/REPrompt/15.pkl \

###

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/MNLIPromptRoberta/15.pkl \


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/WNLIPromptRoberta/15.pkl \


#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/MRPCPromptRoberta/15.pkl \


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/QNLIPromptRoberta/15.pkl \



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/QQPPromptRoberta/15.pkl \


#STSB
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint /data3/private/suyusheng/prompt/prompt/model/STSBPromptRoberta/15.pkl \
'''
