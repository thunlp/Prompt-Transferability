gpus=5
#CUDA_VISIBLE_DEVICES=$gpus

#SST
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/SST2PromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/SST2PromptRoberta/15.pkl \


#RTE
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/RTEPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/RTE_BERT/15.pkl \


#RE
CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/REPrompt.config \
    --gpu $gpus \
    --checkpoint model/REPrompt/15.pkl \
