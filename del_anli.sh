gpus=1

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/anliPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/anliPromptRoberta
