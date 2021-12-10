gpus=1

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticeuseethicsdeontologyPromptRobertaLarge.config \
    --gpu $gpus \
    --checkpoint model/ethicsjusticeuseethicsdeontologyPromptRobertaLarge
