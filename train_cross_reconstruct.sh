gpus=0
MODEL=Roberta
#PROJECTOR=cross_Bert_to_Roberta_reconstructionLoss_all

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross_reconstruct.py --config config/train_task_projection_reconstructionLoss.config \
    --gpu $gpus \
    --target_model $MODEL
    #--projector $PROJECTOR


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross_reconstruct.py --config config/train_task_projection_reconstructionLoss.config \
    --gpu $gpus \
    --target_model $MODEL \
    --mlm
    #--projector $PROJECTOR
'''
