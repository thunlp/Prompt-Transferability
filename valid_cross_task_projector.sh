#CUDA_VISIBLE_DEVICES=$gpus

gpus=3

#MODEL_PROMPT="Roberta-base"
#MODEL_PROMPT="Bert-base"
#MODEL_PROMPT="Random"
#PROJECTOR="model/cross_Bert_to_Roberta_reconstructionLoss"


#restaurant
#FROM_MODEL="Bert"
#TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta_restaurant/13_model_cross_0.516.pkl"
#PROJECTOR="model/crossPromptRoberta_restaurant_100/21_model_cross_0.521.pkl"

#emotion
#FROM_MODEL="Bert"
#TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta_emotion/13_model_cross_0.443.pkl"
#PROJECTOR="model/crossPromptRoberta_emotion_100/7_model_cross_0.485.pkl"
#PROJECTOR="random"

#nli
#FROM_MODEL="Bert"
#TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta_nli/2_model_cross_0.698.pkl"
#PROJECTOR="model/crossPromptRoberta_nli_100/8_model_cross_0.435.pkl"

#all
#FROM_MODEL="Bert"
#TO_MODEL="Roberta"
#PROJECTOR="model/crossPromptRoberta_all/6_model_cross_0.81.pkl"
#PROJECTOR="model/crossPromptRoberta_all_100/30_model_cross_0.541.pkl"
#PROJECTOR="random"





#emotion-large
FROM_MODEL="Roberta"
TO_MODEL="RobertaLarge"
PROJECTOR="model/crossPromptRobertaLarge_emotion_100/114_model_cross_0.691.pkl"
#PROJECTOR="random"


#restaurant-large
#FROM_MODEL="Roberta"
#TO_MODEL="RobertaLarge"
#PROJECTOR="model/crossPromptRobertaLarge_emotion_100/34_model_cross_0.746.pkl"
#PROJECTOR="random"


#nli_large
#FROM_MODEL="Roberta"
#TO_MODEL="RobertaLarge"
#PROJECTOR="model/crossPromptRobertaLarge_nli_100/6_model_cross_0.384.pkl"

#all_large
#FROM_MODEL="Roberta"
#TO_MODEL="RobertaLarge"
#PROJECTOR="model/crossPromptRobertaLarge_all_100/82_model_cross_0.821.pkl"
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




