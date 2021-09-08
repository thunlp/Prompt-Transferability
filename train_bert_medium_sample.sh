mkdir BertMediumForMaskedLM
gpus=2

for DATASET in restaurant
do
    CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/${DATASET}PromptBertMedium.config \
    --gpu $gpus
done
