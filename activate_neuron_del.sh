gpus=3

'''
for MODEL in 'laptop_126' 'IMDB_126' 'tweet_126' 'restaurant_326' 'MNLI_326' 'restaurant_126' 'QQP_326' 'SST2_326' 'SST2_86' 'laptop_326' 'restaurant_86' 'QNLI_326' 'QQP_86' 'IMDB_86' 'snli_326' 'MNLI_86' 'snli_126' 'IMDB_326' 'MNLI_126' 'MRPC_326' 'QNLI_126' 'MRPC_86' 'tweet_86' 'QQP_126' 'laptop_86' 'tweet_326' 'QNLI_86' 'snli_86' 'SST2_126' 'MRPC_126'
do
    echo $MODEL

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron_del.py --config config/activate_neuronPromptRoberta.config \
        --gpu $gpus \
        --replacing_prompt 10_12_newest_randomseed_base/$MODEL \
        --activate_neuron
done
'''






for i in {0,3,6,9}
do
    #python3 activate_neuron_everylayer_sim_del.py $i
    python3 activate_neuron_projector_sim.py $i

    #echo $i
done




