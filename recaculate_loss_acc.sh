#mkdir RobertaForMaskedLM
#mkdir RobertaLargeForMaskedLM
gpus=5


for MODEL in IMDB laptop restaurant MNLI snli ethicsdeontology ethicsjustice SST2 movierationales tweetevalsentiment QNLI recastner QQP MRPC
do
    CUDA_VISIBLE_DEVICES=$gpus python3 recaculate_loss_acc.py --config config/restaurantPromptRoberta.config \
        --gpu $gpus
done

