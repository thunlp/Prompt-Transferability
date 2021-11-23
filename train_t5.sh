TOKENIZERS_PARALLELISM=false

mkdir T5ForMaskedLM
gpus=2



################################
###########T5##############
################################
############
#Sentiment
############

'''
#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/IMDBPromptT5.config \
    --gpu $gpus \



#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptT5.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/laptopPromptT5.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/restaurantPromptT5.config \
    --gpu $gpus
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/movierationalesPromptT5.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/tweetevalsentimentPromptT5.config \
    --gpu $gpus



############
#NLI
############


#MNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MNLIPromptT5.config \
    --gpu $gpus \

#QNLI
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QNLIPromptT5.config \
    --gpu $gpus \


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/snliPromptT5.config \
    --gpu $gpus


############
#EJ
############

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsdeontologyPromptT5.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticePromptT5.config \
    --gpu $gpus




############
#Paraphrase
############

#MRPC
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/MRPCPromptT5.config \
    --gpu $gpus \


#QQP
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/QQPPromptT5.config \
    --gpu $gpus \
'''



#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/SST2PromptT5.config \
    --gpu $gpus \
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
