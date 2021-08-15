gpus=6



#Roberta prompt replaces with mlm-prompt
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta
do
    for PROMPT in IMDBPromptRoberta_mlm laptopPromptRoberta_mlm MNLIPromptRoberta_mlm MRPCPromptRoberta_mlm QNLIPromptRoberta_mlm QQPPromptRoberta_mlm restaurantPromptRoberta_mlm RTEPromptRoberta_mlm SST2PromptRoberta_mlm STSBPromptRoberta_mlm WNLIPromptRoberta_mlm
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/15.pkl
            echo "replace with: " $PROMPT " prompt"
            echo "==========================="

            #Eval mlm
            CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                --gpu $gpus \
                --checkpoint model/$MODEL/15.pkl \
                --replacing_prompt $PROMPT
    done
done


exit



#Bert
'''
for MODEL in IMDBPromptBert laptopPromptBert MNLIPromptBert MRPCPromptBert QNLIPromptBert QQPPromptBert restaurantPromptBert RTEPromptBert SST2PromptBert STSBPromptBert WNLIPromptBert
do
    for PROMPT in IMDBPromptBert laptopPromptBert MNLIPromptBert MRPCPromptBert QNLIPromptBert QQPPromptBert restaurantPromptBert RTEPromptBert SST2PromptBert STSBPromptBert WNLIPromptBert
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/15.pkl
            echo $PROMPT
            echo "==========================="

            #Eval mlm
            CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                --gpu $gpus \
                --checkpoint model/$MODEL/15.pkl \
                --replacing_prompt $PROMPT
    done
done
'''






#Roberta
'''
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta
do
    for PROMPT in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/15.pkl
            echo $PROMPT
            echo "==========================="

            #Eval mlm
            CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                --gpu $gpus \
                --checkpoint model/$MODEL/15.pkl \
                --replacing_prompt $PROMPT
    done
done
'''
