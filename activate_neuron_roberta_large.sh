gpus=0


'''
#for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta WNLIPromptRoberta anliPromptRoberta snliPromptRoberta tweetevalsentimentPromptRoberta movierationalesPromptRoberta recastfactualityPromptRoberta recastpunsPromptRoberta recastverbnetPromptRoberta recastverbcornerPromptRoberta recastnerPromptRoberta recastsentimentPromptRoberta recastmegaveridicalityPromptRoberta ethicscommonsensePromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta ethicsvirtuePromptRoberta


#for MODEL in recastfactualityPromptRoberta recastpunsPromptRoberta recastverbnetPromptRoberta recastverbcornerPromptRoberta recastnerPromptRoberta recastsentimentPromptRoberta recastmegaveridicalityPromptRoberta ethicscommonsensePromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta ethicsvirtuePromptRoberta

for MODEL in IMDBPromptRoberta_label laptopPromptRoberta_label MNLIPromptRoberta_label QNLIPromptRoberta_label QQPPromptRoberta_label restaurantPromptRoberta_label SST2PromptRoberta_label snliPromptRoberta_label tweetevalsentimentPromptRoberta_label movierationalesPromptRoberta_label recastnerPromptRoberta_label ethicsdeontologyPromptRoberta_label ethicsjusticePromptRoberta_label MRPCPromptRoberta_label
'''

#for MODEL in IMDBPromptRobertaLarge
for MODEL in IMDBPromptRobertaLarge laptopPromptRobertaLarge MNLIPromptRobertaLarge QNLIPromptRobertaLarge QQPPromptRobertaLarge restaurantPromptRobertaLarge SST2PromptRobertaLarge snliPromptRobertaLarge tweetevalsentimentPromptRobertaLarge movierationalesPromptRobertaLarge ethicsdeontologyPromptRobertaLarge ethicsjusticePromptRobertaLarge MRPCPromptRobertaLarge
do
    echo "==========================="
    echo activate_neuronPromptRobertaLarge
    echo Replace with task_prompt_emb/$MODEL
    echo "==========================="

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron_roberta_large.py --config config/activate_neuronPromptRobertaLarge.config \
        --gpu $gpus \
        --replacing_prompt task_prompt_emb/$MODEL \
        --activate_neuron
    #exit
done

#--replacing_prompt task_prompt_emb/RobertaLarge_from_chimin/$MODEL \





###
#cd /data/private/suyusheng/prompt/data/activate_neuron_data
#python3 create_valid.py
#cd /data/private/suyusheng/prompt
###

#python3 activate_neuron_sim.py

'''
for i in 9
do
    python3 activate_neuron_everylayer_sim.py $i
    #echo $i
done
'''

