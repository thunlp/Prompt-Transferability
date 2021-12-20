gpus=7

BASEMODEL="T5Small"
#BASEMODEL="T5"
#BASEMODEL="Roberta"
#BASEMODEL="RobertaLarge"
#TARGET
#SOURCE

for DATASET in snliPrompt

#for DATASET in IMDBPrompt laptopPrompt restaurantPrompt SST2Prompt tweetevalsentimentPrompt movierationalesPrompt
#for DATASET in laptopPrompt restaurantPrompt
do
    for PROMPT in IMDBPrompt laptopPrompt MNLIPrompt QNLIPrompt QQPPrompt restaurantPrompt SST2Prompt snliPrompt tweetevalsentimentPrompt movierationalesPrompt ethicsdeontologyPrompt ethicsjusticePrompt MRPCPrompt squadPrompt nq_openPrompt multi_newsPrompt samsumPrompt randomPrompt
    #for PROMPT in IMDBPrompt laptopPrompt MNLIPrompt QNLIPrompt QQPPrompt restaurantPrompt SST2Prompt snliPrompt tweetevalsentimentPrompt movierationalesPrompt ethicsdeontologyPrompt ethicsjusticePrompt MRPCPrompt randomPrompt
    #for PROMPT in IMDBPrompt laptopPrompt restaurantPrompt SST2Prompt tweetevalsentimentPrompt movierationalesPrompt
    #for PROMPT in IMDBPrompt laptopPrompt MNLIPrompt QNLIPrompt QQPPrompt restaurantPrompt SST2Prompt snliPrompt tweetevalsentimentPrompt movierationalesPrompt ethicsdeontologyPrompt ethicsjusticePrompt MRPCPrompt randomPrompt
    do
        echo "==========================="
        echo Model: config/${DATASET}${BASEMODEL}.config
        echo Prompt-emb: task_prompt_emb/${PROMPT}${BASEMODEL}
        echo "==========================="

        CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${DATASET}${BASEMODEL}.config \
            --gpu $gpus \
            --checkpoint model/${DATASET}${BASEMODEL} \
            --replacing_prompt task_prompt_emb/${PROMPT}${BASEMODEL}
    #exit
    done
done




