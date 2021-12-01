gpus=3

BASEMODEL="T5"

for MODEL in IMDBPrompt laptopPrompt MNLIPrompt QNLIPrompt QQPPrompt restaurantPrompt SST2Prompt snliPrompt tweetevalsentimentPrompt movierationalesPrompt ethicsdeontologyPrompt ethicsjusticePrompt MRPCPrompt squadPrompt nq_openPrompt multi_newsPrompt samsumPrompt
#for MODEL in samsumPrompt
do
    echo "==========================="
    echo Model: config/${MODEL}${BASEMODEL}.config
    echo Prompt-emb: task_prompt_emb/${MODEL}${BASEMODEL}
    echo "==========================="

    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${MODEL}${BASEMODEL}.config \
        --gpu $gpus \
        --checkpoint model/${MODEL}${BASEMODEL} \
        --replacing_prompt task_prompt_emb/${MODEL}${BASEMODEL}
done




