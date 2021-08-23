mkdir RobertaForMaskedLM
gpus=4,5,6
nodes=3


################################
###########Roberta##############
################################

############
#Sentiment
############
#restaurant
CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
    --nproc_per_node=$nodes \
    train.py \
    --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    #--local_rank 0
    #--checkpoint roberta-base \
    #--do_test \
    #--comment \
    #--seed

exit


'''
CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
    --nproc_per_node=$nodes \
    train.py --config config/restaurantPromptRoberta.config --gpu $gpus
'''
