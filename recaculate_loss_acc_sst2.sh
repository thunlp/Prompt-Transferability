#mkdir RobertaForMaskedLM
#mkdir RobertaLargeForMaskedLM
gpus=3


for MODEL in SST2
do
    CUDA_VISIBLE_DEVICES=$gpus python3 recaculate_loss_acc.py --config config/${MODEL}PromptRoberta.config \
        --gpu $gpus
done

