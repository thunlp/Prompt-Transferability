#mkdir RobertaForMaskedLM
#mkdir RobertaLargeForMaskedLM
gpus=4


for MODEL in SST2 movierationales tweetevalsentiment QNLI recastner QQP MRPC
do
    CUDA_VISIBLE_DEVICES=$gpus python3 recaculate_loss_acc.py --config config/restaurantPromptRoberta.config \
        --gpu $gpus
done

