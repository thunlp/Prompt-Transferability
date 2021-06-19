#rm -rf task_prompt_emb/*

'''
echo Do you wanna rewrite task emb in the task_prompt_emb y/n ?
read ans

if [ $ans = "y" ]
then
    rm -rf task_prompt_emb/*
else
    echo "Do not rewrite"
    #exit
fi
'''


gpus=2
#CUDA_VISIBLE_DEVICES=$gpus

'''
#SST
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta \
    --return_or_save return
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
exit


#RTE
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTE_BERT \
    --return_or_save return


#RE
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt \

###

#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta \
    --return_or_save return


#WNLI
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta \
    --return_or_save return

'''

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta \
    --return_or_save return


'''
#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta \
    --return_or_save return



#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta \
    --return_or_save return


#STSB
CUDA_VISIBLE_DEVICES=$gpus python3 check_prompt_randomseed.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta \
    --return_or_save return
'''
