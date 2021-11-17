gpus=7


#for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta SST2PromptRoberta snliPromptRoberta tweetevalsentimentPromptRoberta movierationalesPromptRoberta recastnerPromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta MRPCPromptRoberta
#for MODEL in IMDB_base_emotionPromptRoberta MNLI_base_nliPromptRoberta laptop_base_emotionPromptRoberta laptop_base_nliPromptRoberta restaurant_base_emotionPromptRoberta restaurant_base_nliPromptRoberta snli_base_emotionPromptRoberta snli_base_nliPromptRoberta IMDB_base_nliPromptRoberta MNLI_base_emotionPromptRoberta
#for MODEL in IMDB_base_nliPromptRoberta MNLI_base_emotionPromptRoberta IMDB_base_emotionPromptRoberta MNLI_base_nliPromptRoberta

for MODEL in IMDBPromptRoberta_proj laptopPromptRoberta_proj restaurantPromptRoberta_proj SST2PromptRoberta_proj movierationalesPromptRoberta_proj tweetevalsentimentPromptRoberta_proj MNLIPromptRoberta_proj QNLIPromptRoberta_proj snliPromptRoberta_proj
#for MODEL in MNLI_trainedPromptRoberta QNLI_trainedPromptRoberta SST2_trainedPromptRoberta laptop_trainedPromptRoberta movierationales_trainedPromptRoberta restaurant_trainedPromptRoberta snli_trainedPromptRoberta tweetevalsentiment_trainedPromptRoberta IMDB_trainedPromptRoberta
do
    echo "==========================="
    echo activate_neuronPromptRoberta
    echo Replace with task_prompt_emb/$MODEL
    echo "==========================="

    #Eval mlm
    #CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron_plm.py --config config/activate_neuronPromptRoberta.config \
    #CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron_ffn.py --config config/activate_neuronPromptRoberta.config \
    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py --config config/activate_neuronPromptRoberta.config \
        --gpu $gpus \
        --replacing_prompt task_prompt_emb/$MODEL \
        --activate_neuron
done





#python3 activate_neuron_everylayer_sim_del.py

#python3 activate_neuron_projector_sim.py

'''
#for i in {0..11}
for i in 11
do
    python3 activate_neuron_projector_everylayer_sim.py $i
done
'''




