gpus=7


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




'''
#Bert
for MODEL in IMDBPromptBert laptopPromptBert MNLIPromptBert MRPCPromptBert QNLIPromptBert QQPPromptBert restaurantPromptBert RTEPromptBert SST2PromptBert STSBPromptBert WNLIPromptBert anliPromptBert emobankarousalPromptBert emobankdominancePromptBert movierationalesPromptBert tweetevalsentimentPromptBert persuasivenessrelevancePromptBert persuasivenessspecificityPromptBert snliPromptBert squinkyformalityPromptBert squinkyimplicaturePromptBert
do
    for PROMPT in IMDBPromptBert laptopPromptBert MNLIPromptBert MRPCPromptBert QNLIPromptBert QQPPromptBert restaurantPromptBert RTEPromptBert SST2PromptBert STSBPromptBert WNLIPromptBert anliPromptBert emobankarousalPromptBert emobankdominancePromptBert movierationalesPromptBert tweetevalsentimentPromptBert persuasivenessrelevancePromptBert persuasivenessspecificityPromptBert snliPromptBert squinkyformalityPromptBert squinkyimplicaturePromptBert
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL
            echo task_prompt_emb/$PROMPT
            echo "==========================="

            if [ $MODEL == anliPromptBert ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt task_prompt_emb/$PROMPT
            elif [ $MODEL == snliPromptBert ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt task_prompt_emb/$PROMPT
            else
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt task_prompt_emb/$PROMPT
            fi
    done
done
'''





'''
#Roberta
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta anliPromptRoberta emobankarousalPromptRoberta emobankdominancePromptRoberta movierationalesPromptRoberta tweetevalsentimentPromptRoberta persuasivenessrelevancePromptRoberta persuasivenessspecificityPromptRoberta snliPromptRoberta squinkyformalityPromptRoberta squinkyimplicaturePromptRoberta
do
    for PROMPT in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta anliPromptRoberta emobankarousalPromptRoberta emobankdominancePromptRoberta movierationalesPromptRoberta tweetevalsentimentPromptRoberta persuasivenessrelevancePromptRoberta persuasivenessspecificityPromptRoberta snliPromptRoberta squinkyformalityPromptRoberta squinkyimplicaturePromptRoberta
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/
            echo task_prompt_emb/$PROMPT
            echo "==========================="

            #Eval mlm
            if [ $MODEL == anliPromptRoberta ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt task_prompt_emb/$PROMPT
            elif [$MODEL == snliPromptRoberta ]
            then
                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt task_prompt_emb/$PROMPT
            else
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                    --gpu $gpus \
                    --checkpoint model/$MODEL \
                    --replacing_prompt task_prompt_emb/$PROMPT
            fi
    done
done
'''


#Roberta
'''
#for MODEL in QQPPromptRoberta
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta WNLIPromptRoberta anliPromptRoberta movierationalesPromptRoberta tweetevalsentimentPromptRoberta snliPromptRoberta recastfactualityPromptRoberta recastpunsPromptRoberta recastverbcornerPromptRoberta recastnerPromptRoberta recastsentimentPromptRoberta recastmegaveridicalityPromptRoberta ethicscommonsensePromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta
do
    for PROMPT in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta WNLIPromptRoberta anliPromptRoberta movierationalesPromptRoberta tweetevalsentimentPromptRoberta snliPromptRoberta recastfactualityPromptRoberta recastpunsPromptRoberta recastverbcornerPromptRoberta recastnerPromptRoberta recastsentimentPromptRoberta recastmegaveridicalityPromptRoberta ethicscommonsensePromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta
    #for PROMPT in recastnerPromptRoberta recastpunsPromptRoberta recastverbcornerPromptRoberta recastfactualityPromptRoberta recastmegaveridicalityPromptRoberta recastsentimentPromptRoberta
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/
            echo task_prompt_emb/$PROMPT
            echo "==========================="

            CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                --gpu $gpus \
                --checkpoint model/$MODEL \
                --replacing_prompt task_prompt_emb/$PROMPT
    done
done
'''



#Roberta_label
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta SST2PromptRoberta movierationalesPromptRoberta tweetevalsentimentPromptRoberta snliPromptRoberta recastnerPromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta
do
    for PROMPT in IMDBPromptRoberta_label laptopPromptRoberta_label MNLIPromptRoberta_label QNLIPromptRoberta_label QQPPromptRoberta_label restaurantPromptRoberta_label SST2PromptRoberta_label movierationalesPromptRoberta_label tweetevalsentimentPromptRoberta_label snliPromptRoberta_label recastnerPromptRoberta_label ethicsdeontologyPromptRoberta_label ethicsjusticePromptRoberta_label
    do
            echo "==========================="
            echo config/$MODEL.config
            echo model/$MODEL/
            echo task_prompt_emb/$PROMPT
            echo "==========================="

            CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL.config \
                --gpu $gpus \
                --checkpoint model/$MODEL \
                --replacing_prompt task_prompt_emb/$PROMPT
    done
done
