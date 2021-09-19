gpus=4

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/ethicsjusticePromptRoberta_label.config \
    --gpu $gpus \
    --checkpoint model/ethicsjusticePromptRoberta_label
