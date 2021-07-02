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
############
#Sentiment Classification
############

#restaurant
rm -rf task_prompt_emb/IMDBPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta/15.pkl \
    --return_or_save save

#restaurant
rm -rf task_prompt_emb/restaurantPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta/15.pkl \
    --return_or_save save

#laptop
rm -rf task_prompt_emb/laptopPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta/15.pkl \
    --return_or_save save

#SST
rm -rf task_prompt_emb/SST2PromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta/15.pkl \
    --return_or_save save
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed

############
#NLI
############

#RTE
rm -rf task_prompt_emb/RTEPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta/15.pkl \
    --return_or_save save

#MNLI
rm -rf task_prompt_emb/MNLIPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta/13.pkl \
    --return_or_save save

#WNLI
rm -rf task_prompt_emb/WNLIPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta/15.pkl \
    --return_or_save save

############
#Paraphrase
############

#MRPC
rm -rf task_prompt_emb/MRPCPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta/15.pkl \
    --return_or_save save


#QQP
rm -rf task_prompt_emb/QQPPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta/15.pkl \
    --return_or_save save




############
#RE
############
#RE
rm -rf task_prompt_emb/REPrompt
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt/15.pkl \





############
#Other
############
#QNLI
rm -rf task_prompt_emb/QNLIPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta/15.pkl \
    --return_or_save save


#STSB
rm -rf task_prompt_emb/STSBPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta/15.pkl \
    --return_or_save save
