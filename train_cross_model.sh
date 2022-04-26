gpus=7


###########################################
#laptop: from T5Base to RobertaBase########
###########################################

DATASET="laptop"

#SOURCE_PROMPT
#MODEL_PROMPT="Bert-base"
#MODEL_PROMPT="Roberta-base"
MODEL_PROMPT="T5-base"

#SOURCE_MODEL to TARGET_MODEL
SOURCE_MODEL="T5"
TARGET_MODEL="Roberta"
source_model="t5"
target_model="roberta"

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPrompt${TARGET_MODEL}_${DATASET}_100_${source_model}_to_${target_model}.config \
    --gpu $gpus \
    --model_prompt $MODEL_PROMPT


#####################################################
#restaurant: from RobertaBase to RobertaLarge########
#####################################################
DATASET="restaurant"
#DATASET="MNLI"

#SOURCE_PROMPT
#MODEL_PROMPT="Bert-base"
MODEL_PROMPT="Roberta-base"

#SOURCE_MODEL to TARGET_MODEL
SOURCE_MODEL="RobertaBase"
TARGET_MODEL="RobertaLarge"

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPrompt${TARGET_MODEL}_${DATASET}_100.config \
    --gpu $gpus \
    --model_prompt ${MODEL_PROMPT}



###########################################
#MNLI: from BertBase to RobertaBase########
###########################################
DATASET="MNLI"

#SOURCE_PROMPT
MODEL_PROMPT="Bert-base"
#MODEL_PROMPT="Roberta-base"
#MODEL_PROMPT="T5-base"

#SOURCE_MODEL to TARGET_MODEL
SOURCE_MODEL="Bert"
TARGET_MODEL="Roberta"
source_model="Bert"
target_model="Roberta"

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPrompt${TARGET_MODEL}_${DATASET}_100.config \
    --gpu $gpus \
    --model_prompt ${MODEL_PROMPT}




############################################################
#MNLI: from BertBase to RobertaBase (With LayerNorm)########
############################################################
#DATASET="restaurant"
DATASET="nli"

#SOURCE_PROMPT
MODEL_PROMPT="Bert-base"
#MODEL_PROMPT="Roberta-base"

#SOURCE_MODEL to TARGET_MODEL
SOURCE_MODEL="BertBase"
TARGET_MODEL="RobertaBase"

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPrompt${TARGET_MODEL}_${DATASET}_100.config \
    --gpu $gpus \
    --model_prompt ${MODEL_PROMPT}





###########################################
#All task: from BertBase to RobertaBase####
###########################################
DATASET="all"

MODEL_PROMPT="Bert-base"

#SOURCE_MODEL to TARGET_MODEL
SOURCE_MODEL="Bert"
TARGET_MODEL="Roberta"
source_model="Bert"
target_model="Roberta"

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross.py --config config/crossPrompt${TARGET_MODEL}_${DATASET}_100.config \
    --gpu $gpus \
    --model_prompt ${MODEL_PROMPT}
