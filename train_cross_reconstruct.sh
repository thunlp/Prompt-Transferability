gpus=0
MODEL=Roberta

CUDA_VISIBLE_DEVICES=$gpus python3 train_cross_reconstruct.py --config config/train_task_projection_reconstructionLoss.config \
    --gpu $gpus \
    --target_model $MODEL


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train_cross_reconstruct.py --config config/train_task_projection_reconstructionLoss.config \
    --gpu $gpus \
    --target_model $MODEL \
    --mlm
'''
