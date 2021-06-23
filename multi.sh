
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 multi.py

CUDA_VISIBLE_DEVICES=3,7,2,5 python -m torch.distributed.launch --nproc_per_node=4 multi.py
