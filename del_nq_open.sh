gpus=4


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/nq_openPromptT5.config \
    --gpu $gpus \
    --checkpoint model/nq_openPromptT5
