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
    --checkpoint model/SST2PromptRoberta/10.pkl \
    --return_or_save save
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#RTE
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTE_BERT/15.pkl \
    --return_or_save save


#RE
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt/15.pkl \

###

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta/13.pkl \
    --return_or_save save


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta/15.pkl \
    --return_or_save save


#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta/15.pkl \
    --return_or_save save


#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta/15.pkl \
    --return_or_save save



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta/15.pkl \
    --return_or_save save


#STSB
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta/15.pkl \
    --return_or_save save
