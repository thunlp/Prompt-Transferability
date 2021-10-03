#mkdir RobertaForMaskedLM
#mkdir RobertaLargeForMaskedLM
gpus=2


for MODEL in IMDB laptop restaurant MNLI
do
    CUDA_VISIBLE_DEVICES=$gpus python3 recaculate_loss_acc.py --config config/${MODEL}PromptRoberta.config \
        --gpu $gpus
done

