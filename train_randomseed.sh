mkdir RobertaForMaskedLM
gpus=0
#nproc_per_node_num=2


#SST-2
#CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node_num train_randomseed.py --config config/SST2PromptRoberta.config \

for i in {1..8}
do


    #MRPC
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/MRPCPromptRoberta.config \
        --gpu $gpus \
        --seed $i


    '''
    #SST2
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/SST2PromptRoberta.config \
        --gpu $gpus \
        --seed $i
        #--distributed \
        #--backend nccl
        #--local_rank $gpus \
        #--checkpoint roberta-base \
        #--do_test \
        #--comment \


    #RTE
    #Remove prompts between two sentences
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/RTEPromptRoberta.config \
        --gpu $gpus \
        --seed $i


    #RE
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/REPrompt.config \
        --gpu $gpus \
        --seed $i


    #MNLI
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/MNLIPromptRoberta.config \
        --gpu $gpus \
        --seed $i


    #WNLI
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/WNLIPromptRoberta.config \
        --gpu $gpus \
        --seed $i


    #QNLI
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/QNLIPromptRoberta.config \
        --gpu $gpus \
        --seed $i


    #QQP
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/QQPPromptRoberta.config \
        --gpu $gpus \
        --seed $i


    #STSB
    CUDA_VISIBLE_DEVICES=$gpus python3 train_randomseed.py --config config/STSBPromptRoberta.config \
        --gpu $gpus \
        --seed $i
    '''

done

