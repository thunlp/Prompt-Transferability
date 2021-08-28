gpus=1
MODEL=Roberta

CUDA_VISIBLE_DEVICES=$gpus python3 train_task_projection_reconstructionLoss.py --config config/train_task_projection_reconstructionLoss.config \
    --gpu $gpus \
    --target_model $MODEL
