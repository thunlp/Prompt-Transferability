gpus=6


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
