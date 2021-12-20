gpus=1


CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/nq_openPromptT5Small.config \
    --gpu $gpus \
    --checkpoint model/nq_openPromptT5Small
