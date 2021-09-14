gpus=7

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsvirtuePromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/ethicsvirtuePromptRoberta
