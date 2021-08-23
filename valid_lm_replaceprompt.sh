gpus=3


for MODEL in agnewsPromptRoberta cs_wikiPromptRoberta IMDBPromptRoberta sciercPromptRoberta SST2PromptRoberta
do
    for S_1 in "s1" "s2"
    do
        for PROMPT in agnewsPromptRoberta cs_wikiPromptRoberta IMDBPromptRoberta sciercPromptRoberta SST2PromptRoberta
        do
            for S_2 in "s1" "s2"
            do
                echo "==========================="
                echo config/$MODEL"_"$S_1".config"
                echo model/$MODEL"_"$S_1/31.pkl
                echo $PROMPT"_"$S_2
                echo "==========================="

                #Eval mlm
                CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/$MODEL"_"$S_1".config" \
                    --gpu $gpus \
                    --checkpoint model/$MODEL"_mlm_"$S_1/31.pkl \
                    --replacing_prompt $PROMPT"_mlm_"$S_2 \
                    --pre_train_mlm

            done
        done
    done
done
