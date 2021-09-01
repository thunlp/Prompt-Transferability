gpus=5


'''
#Roberta prompt replaces with mlm-prompt
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta
do
    for PROMPT in IMDBPromptRoberta_mlm laptopPromptRoberta_mlm MRPCPromptRoberta_mlm restaurantPromptRoberta_mlm SST2PromptRoberta_mlm
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/1.pkl
            echo "replace with: " $PROMPT " prompt"
            echo "==========================="

            #Eval mlm
            CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                --gpu $gpus \
                --checkpoint model/$MODEL/1.pkl \
                --replacing_prompt $PROMPT
    done
done
'''





#Bert
for MODEL in IMDBPromptBert laptopPromptBert MNLIPromptBert MRPCPromptBert QNLIPromptBert QQPPromptBert restaurantPromptBert RTEPromptBert SST2PromptBert STSBPromptBert WNLIPromptBert anliPromptBert emobankarousalPromptBert emobankdominancePromptBert movierationalesPromptBert tweetevalsentimentPromptBert persuasivenessrelevancePromptBert persuasivenessspecificityPromptBert snliPromptBert squinkyformalityPromptBert squinkyimplicaturePromptBert
do
    for PROMPT in IMDBPromptBert laptopPromptBert MNLIPromptBert MRPCPromptBert QNLIPromptBert QQPPromptBert restaurantPromptBert RTEPromptBert SST2PromptBert STSBPromptBert WNLIPromptBert anliPromptBert emobankarousalPromptBert emobankdominancePromptBert movierationalesPromptBert tweetevalsentimentPromptBert persuasivenessrelevancePromptBert persuasivenessspecificityPromptBert snliPromptBert squinkyformalityPromptBert squinkyimplicaturePromptBert
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL
            echo $PROMPT
            echo "==========================="

            if [ $MODEL == anliPromptBert ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt model/$PROMPT
            elif [ $MODEL == snliPromptBert ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt model/$PROMPT
            else
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt model/$PROMPT
            fi
    done
done






#Roberta
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta anliPromptRoberta emobankarousalPromptRoberta emobankdominancePromptRoberta movierationalesPromptRoberta tweetevalsentimentPromptRoberta persuasivenessrelevancePromptRoberta persuasivenessspecificityPromptRoberta snliPromptRoberta squinkyformalityPromptRoberta squinkyimplicaturePromptRoberta
do
    for PROMPT in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta anliPromptRoberta emobankarousalPromptRoberta emobankdominancePromptRoberta movierationalesPromptRoberta tweetevalsentimentPromptRoberta persuasivenessrelevancePromptRoberta persuasivenessspecificityPromptRoberta snliPromptRoberta squinkyformalityPromptRoberta squinkyimplicaturePromptRoberta
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/
            echo $PROMPT
            echo "==========================="

            #Eval mlm
            if [ $MODEL == anliPromptRoberta ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt model/$PROMPT
            elif [$MODEL == snliPromptRoberta ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt model/$PROMPT
            else
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt model/$PROMPT
            fi
    done
done
