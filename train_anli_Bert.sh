gpus=5

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/anliPromptBert.config \
    --checkpoint model/anliPromptBert/15.pkl \
    --gpu $gpus
