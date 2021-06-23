mkdir RobertaForMaskedLM
mkdir RobertaLargeForMaskedLM
gpus=3,7
NUM_GPUS=2




CUDA_VISIBLE_DEVICES=$gpus python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_multiGPU.py --config config/mutiGPU_STSBPromptRoberta.config \
    --gpu $gpus \
    --local_rank 0 \


'''
CUDA_VISIBLE_DEVICES=$gpus python3 train_multiGPU.py --config config/mutiGPU_STSBPromptRoberta.config \
    --gpu $gpus \
'''


