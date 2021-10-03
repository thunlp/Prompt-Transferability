#mkdir RobertaForMaskedLM
#mkdir RobertaLargeForMaskedLM
gpus=5


for MODEL in snli ethicsdeontology ethicsjustice
do
    CUDA_VISIBLE_DEVICES=$gpus python3 recaculate_loss_acc.py --config config/${MODEL}PromptRoberta.config \
        --gpu $gpus
done

