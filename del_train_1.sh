gpus=0

#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed




#laptop
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/laptopPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed



#IMDB
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/IMDBPromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True



#SST-2
CUDA_VISIBLE_DEVICES=$gpus python3 train_lm.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --pre_train_mlm True
    #--checkpoint roberta-base \
    #--local_rank \
    #--do_test \
    #--comment \
    #--seed
