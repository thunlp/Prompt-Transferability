#rm -rf task_prompt_emb/*

gpus=6

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


#CUDA_VISIBLE_DEVICES=$gpus
############
#Sentiment Classification
############
#IMDB
rm -rf task_prompt_emb/IMDBPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta/15.pkl \
    --mode extract_prompt

#restaurant
rm -rf task_prompt_emb/restaurantPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta/15.pkl \
    --mode extract_prompt



#laptop
rm -rf task_prompt_emb/laptopPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta/15.pkl \
    --mode extract_prompt



#SST
rm -rf task_prompt_emb/SST2PromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta/15.pkl \
    --mode extract_prompt
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
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta/15.pkl \
    --mode extract_prompt

#MNLI
rm -rf task_prompt_emb/MNLIPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta/15.pkl \
    --mode extract_prompt

#WNLI
rm -rf task_prompt_emb/WNLIPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta/15.pkl \
    --mode extract_prompt

############
#Paraphrase
############

#MRPC
rm -rf task_prompt_emb/MRPCPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta/15.pkl \
    --mode extract_prompt


#QQP
rm -rf task_prompt_emb/QQPPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta/15.pkl \
    --mode extract_prompt




############
#RE
############
#RE
rm -rf task_prompt_emb/REPrompt
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt/15.pkl \




############
#Other
############
#QNLI
rm -rf task_prompt_emb/QNLIPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta/15.pkl \
    --mode extract_prompt


#STSB
rm -rf task_prompt_emb/STSBPromptRoberta
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta/15.pkl \
    --mode extract_prompt
'''





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
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert/15.pkl \
    --mode extract_prompt


#restaurant
rm -rf task_prompt_emb/restaurantPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptBert/15.pkl \
    --mode extract_prompt

#laptop
rm -rf task_prompt_emb/laptopPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptBert/15.pkl \
    --mode extract_prompt


#SST
rm -rf task_prompt_emb/SST2PromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert/15.pkl \
    --mode extract_prompt
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
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptBert.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptBert/15.pkl \
    --mode extract_prompt

#MNLI
rm -rf task_prompt_emb/MNLIPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptBert/15.pkl \
    --mode extract_prompt

#WNLI
rm -rf task_prompt_emb/WNLIPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptBert/15.pkl \
    --mode extract_prompt

############
#Paraphrase
############

#MRPC
rm -rf task_prompt_emb/MRPCPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptBert.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptBert/15.pkl \
    --mode extract_prompt


#QQP
rm -rf task_prompt_emb/QQPPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptBert/15.pkl \
    --mode extract_prompt




############
#RE
############
#RE
rm -rf task_prompt_emb/REPrompt
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt/15.pkl \





############
#Other
############
#QNLI
rm -rf task_prompt_emb/QNLIPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptBert.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptBert/15.pkl \
    --mode extract_prompt


#STSB
rm -rf task_prompt_emb/STSBPromptBert
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptBert.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptBert/15.pkl \
    --mode extract_prompt
'''




'''
################
######MLM
################

#IMDB
rm -rf task_prompt_emb/IMDBPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta_mlm/15.pkl \
    --mode extract_prompt

#restaurant
rm -rf task_prompt_emb/restaurantPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta_mlm/15.pkl \
    --mode extract_prompt

#laptop
rm -rf task_prompt_emb/laptopPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/laptopPromptRoberta_mlm/15.pkl \
    --mode extract_prompt

#SST
rm -rf task_prompt_emb/SST2PromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta_mlm/15.pkl \
    --mode extract_prompt
    #--result /data3/private/suyusheng/prompt/prompt/model/SST2PromptRoberta_mlm_result \
    #--data_type eval \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


############
#Paraphrase
############

#MRPC
rm -rf task_prompt_emb/MRPCPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MRPCPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MRPCPromptRoberta_mlm/15.pkl \
    --mode extract_prompt

exit

#QQP
rm -rf task_prompt_emb/QQPPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QQPPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QQPPromptRoberta_mlm/15.pkl \
    --mode extract_prompt


############
#NLI
############

#RTE
rm -rf task_prompt_emb/RTEPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTEPromptRoberta_mlm/15.pkl \
    --mode extract_prompt

#MNLI
rm -rf task_prompt_emb/MNLIPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/MNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/MNLIPromptRoberta_mlm/15.pkl \
    --mode extract_prompt

#WNLI
rm -rf task_prompt_emb/WNLIPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/WNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/WNLIPromptRoberta_mlm/15.pkl \
    --mode extract_prompt




############
#RE
############
#RE
rm -rf task_prompt_emb/REPrompt
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt_mlm/15.pkl \
    --mode extract_prompt





############
#Other
############
#QNLI
rm -rf task_prompt_emb/QNLIPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/QNLIPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/QNLIPromptRoberta_mlm/15.pkl \
    --mode extract_prompt


#STSB
rm -rf task_prompt_emb/STSBPromptRoberta_mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/STSBPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/STSBPromptRoberta_mlm/15.pkl \
    --mode extract_prompt
'''



######################
######################
######################
#Extract prompt mlm
'''
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/agnewsPromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptRoberta_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/agnewsPromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptRoberta_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/cs_wikiPromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptRoberta_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm


#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/cs_wikiPromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptRoberta_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm


#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptRoberta_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/sciercPromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/sciercPromptRoberta_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm


#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/sciercPromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/sciercPromptRoberta_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta_s1.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm

#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta_s2.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm
'''




#Berta mlm
'''
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/agnewsPromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptBert_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm
'''



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/agnewsPromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/agnewsPromptBert_mlm_s2/4.pkl \
    --mode extract_prompt \
    --pre_train_mlm

exit


#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/cs_wikiPromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptBert_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm


#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/cs_wikiPromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/cs_wikiPromptBert_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm


#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/IMDBPromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/IMDBPromptBert_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/sciercPromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/sciercPromptBert_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm


#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/sciercPromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/sciercPromptBert_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm



#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptBert_s1.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert_mlm_s1/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm

#Extract prompt mlm
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptBert_s2.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptBert_mlm_s2/15.pkl \
    --mode extract_prompt \
    --pre_train_mlm
