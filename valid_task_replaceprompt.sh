gpus=7

BASEMODEL="T5"

#FROM_MODEL="Roberta"
#TO_MODEL="RobertaLarge"
#PROJECTOR="random"
for MODEL in IMDBPrompt laptopPrompt MNLIPrompt QNLIPrompt QQPPrompt restaurantPrompt SST2Prompt snliPrompt tweetevalsentimentPrompt movierationalesPrompt recastnerPrompt ethicsdeontologyPrompt ethicsjusticePrompt MRPCPrompt
do
    echo "==========================="
    echo Model: config/${MODEL}${BASEMODEL}.config
    echo Prompt-emb: task_prompt_emb/${MODEL}${BASEMODEL}
    echo "==========================="

    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${MODEL}${BASEMODEL}.config \
        --gpu $gpus \
        --checkpoint model/${MODEL}${BASEMODEL} \
        --replacing_prompt task_prompt_emb/${MODEL}${BASEMODEL} \
done




'''
FROM_MODEL="Bert"
TO_MODEL="Roberta"


#Roberta
#for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta STSBPromptRoberta WNLIPromptRoberta
for MODEL in IMDBPrompt laptopPrompt MNLIPrompt QNLIPrompt QQPPrompt restaurantPrompt SST2Prompt snliPrompt tweetevalsentimentPrompt movierationalesPrompt recastnerPrompt ethicsdeontologyPrompt ethicsjusticePrompt MRPCPrompt
do
    echo "==========================="
    echo config/${MODEL}${TO_MODEL}.config
    echo task_prompt_emb/${MODEL}${FROM_MODEL}
    echo "==========================="

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${MODEL}${TO_MODEL}.config \
        --gpu $gpus \
        --checkpoint model/${MODEL}${TO_MODEL} \
        --replacing_prompt task_prompt_emb/${MODEL}${FROM_MODEL}
        #--checkpoint task_prompt_emb/${MODEL}${FROM_MODEL} \
        #--task_transfer_projector
done
'''
