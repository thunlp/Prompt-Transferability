#mkdir RobertaForMaskedLM
#mkdir RobertaLargeForMaskedLM
gpus=7


for MODEL in tweetevalsentiment
do
    CUDA_VISIBLE_DEVICES=$gpus python3 recaculate_loss_acc.py --config config/${MODEL}PromptRoberta.config \
        --gpu $gpus
done

