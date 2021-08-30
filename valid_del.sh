#CUDA_VISIBLE_DEVICES=$gpus

gpus=3

#MODEL_PROMPT="Roberta-base"
#MODEL_PROMPT="Roberta-base"
#MODEL_PROMPT="Random"


############
#Sentiment
############
#restaurant
# Refer to config/restaurantPromptRoberta_large.config
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/restaurantPromptRobertai_large.config \
    --gpu $gpus \
    --checkpoint RobertaLargeForMaskedLM/cross_mlmPrompt/pytorch_model.bin \
    --replacing_prompt restaurantPromptRoberta \
    --model_transfer_projector


