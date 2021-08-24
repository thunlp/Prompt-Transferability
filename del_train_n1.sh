gpus=1,5
nodes=2


CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
    --nproc_per_node=$nodes \
    train.py \
    --config config/snliPromptRoberta.config \
    --gpu $gpus


CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch \
    --nproc_per_node=$nodes \
    train.py \
    --config config/anliPromptRoberta.config \
    --gpu $gpus
