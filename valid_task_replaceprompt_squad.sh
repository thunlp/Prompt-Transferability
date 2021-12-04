gpus=7

BASEMODEL="T5"
#TARGET
#SOURCE

for DATASET in squadPrompt
do
    for PROMPT in movierationalesPrompt
    do
        echo "==========================="
        echo Model: config/${DATASET}${BASEMODEL}.config
        echo Prompt-emb: task_prompt_emb/${PROMPT}${BASEMODEL}
        echo "==========================="

        CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${DATASET}${BASEMODEL}.config \
            --gpu $gpus \
            --checkpoint model/${DATASET}${BASEMODEL} \
            --replacing_prompt task_prompt_emb/${PROMPT}${BASEMODEL}
    done
done




