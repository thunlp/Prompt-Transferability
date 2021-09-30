#CUDA_VISIBLE_DEVICES=$gpus

gpus=5

#MODEL_PROMPT="Roberta-base"
#MODEL_PROMPT="Bert-base"
#MODEL_PROMPT="Random"
#PROJECTOR="model/cross_Bert_to_Roberta_reconstructionLoss"


#emotion
#FROM_MODEL="Bert"
#TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta"
#PROJECTOR="model/crossPromptRoberta_emotion_76800/20_model_cross.pkl"

#emotion
FROM_MODEL="Bert"
TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta"
#PROJECTOR="model/crossPromptRoberta_emotion/13_model_cross_0.443.pkl"
PROJECTOR="model/crossPromptRoberta_emotion_100/7_model_cross_0.485.pkl"
#PROJECTOR="random"

#nli
#FROM_MODEL="Bert"
#TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta_nli/30_model_cross.pkl"

#all
#FROM_MODEL="Bert"
#TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta_all/99_model_cross.pkl"
#PROJECTOR="random"


#emotion-large
#FROM_MODEL="Roberta"
#TO_MODEL="RobertaLarge"
#PROJECTOR="model/crossPromptRobertaLarge_emotion/76_model_cross.pkl"
#PROJECTOR="random"

#all_large
#FROM_MODEL="Roberta"
#TO_MODEL="RobertaLarge"
#PROJECTOR="model/crossPromptRobertaLarge_nli/17_model_cross.pkl"

#all_large
#FROM_MODEL="Roberta"
#TO_MODEL="RobertaLarge"
#PROJECTOR="model/crossPromptRobertaLarge_all/219_model_cross.pkl"
#PROJECTOR="random"



for MODEL in IMDBPrompt laptopPrompt MNLIPrompt QNLIPrompt QQPPrompt restaurantPrompt SST2Prompt snliPrompt tweetevalsentimentPrompt movierationalesPrompt recastnerPrompt ethicsdeontologyPrompt ethicsjusticePrompt MRPCPrompt
#for MODEL in MRPCPrompt
do
    echo "==========================="
    echo Model: config/${MODEL}${TO_MODEL}.config
    echo Prompt-emb: task_prompt_emb/${MODEL}${FROM_MODEL}
    echo "==========================="

    CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${MODEL}${TO_MODEL}.config \
        --gpu $gpus \
        --checkpoint model/${MODEL}${TO_MODEL} \
        --replacing_prompt task_prompt_emb/${MODEL}${FROM_MODEL} \
        --model_transfer_projector \
        --projector $PROJECTOR
done




