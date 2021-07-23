#rm -rf task_prompt_emb/*

gpus=4

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

'''
#CUDA_VISIBLE_DEVICES=$gpus
############
#Sentiment Classification
############
#IMDB
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
    --checkpoint model/MNLIPromptRoberta/15.pkl \
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
'''


####################################
############Bert####################
####################################

#CUDA_VISIBLE_DEVICES=$gpus
############
#Sentiment Classification
############
#IMDB
rm -rf task_prompt_emb/IMDBPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert/15.pkl \
    --return_or_save save


#restaurant
rm -rf task_prompt_emb/restaurantPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert/15.pkl \
    --return_or_save save

#laptop
rm -rf task_prompt_emb/laptopPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert/15.pkl \
    --return_or_save save


#SST
rm -rf task_prompt_emb/SST2PromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert/15.pkl \
    --return_or_save save
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptBert_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


############
#NLI
############

#RTE
rm -rf task_prompt_emb/RTEPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptBert/15.pkl \
    --return_or_save save

#MNLI
rm -rf task_prompt_emb/MNLIPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptBert/15.pkl \
    --return_or_save save

#WNLI
rm -rf task_prompt_emb/WNLIPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptBert/15.pkl \
    --return_or_save save

############
#Paraphrase
############

#MRPC
rm -rf task_prompt_emb/MRPCPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert/15.pkl \
    --return_or_save save


#QQP
rm -rf task_prompt_emb/QQPPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert/15.pkl \
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
rm -rf task_prompt_emb/QNLIPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptBert/15.pkl \
    --return_or_save save


#STSB
rm -rf task_prompt_emb/STSBPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 create.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptBert/15.pkl \
    --return_or_save save

