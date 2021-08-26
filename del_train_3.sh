gpus=5

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/restaurantPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True

    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/laptopPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed


#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True


#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptBert.config \
    --gpu $gpus \
    --pre_train_mlm True
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
