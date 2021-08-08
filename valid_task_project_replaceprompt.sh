gpus=6

'''
#Bert
for MODEL in IMDBPromptBert laptopPromptBert MNLIPromptBert MRPCPromptBert QNLIPromptBert QQPPromptBert restaurantPromptBert RTEPromptBert SST2PromptBert STSBPromptBert WNLIPromptBert
do
    echo "==========================="
    echo config/$MODEL.config
    echo model/$MODEL/15.pkl
    echo "==========================="

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
        --gpu $gpus \
        --checkpoint model/$MODEL/15.pkl \
        --task_transfer_projector
done
'''






#Roberta
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta
do
    echo "==========================="
    echo config/$MODEL.config
    echo model/$MODEL/15.pkl
    echo "==========================="

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
        --gpu $gpus \
        --checkpoint model/$MODEL/15.pkl \
        --task_transfer_projector
    exit
done
