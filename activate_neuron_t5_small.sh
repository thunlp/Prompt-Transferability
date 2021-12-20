gpus=7


'''
#for MODEL in IMDBPromptT5 laptopPromptT5 MNLIPromptT5 MRPCPromptT5 QNLIPromptT5 QQPPromptT5 restaurantPromptT5 RTEPromptT5 SST2PromptT5 WNLIPromptT5 anliPromptT5 snliPromptT5 tweetevalsentimentPromptT5 movierationalesPromptT5 recastfactualityPromptT5 recastpunsPromptT5 recastverbnetPromptT5 recastverbcornerPromptT5 recastnerPromptT5 recastsentimentPromptT5 recastmegaveridicalityPromptT5 ethicscommonsensePromptT5 ethicsdeontologyPromptT5 ethicsjusticePromptT5 ethicsvirtuePromptT5


#for MODEL in recastfactualityPromptT5 recastpunsPromptT5 recastverbnetPromptT5 recastverbcornerPromptT5 recastnerPromptT5 recastsentimentPromptT5 recastmegaveridicalityPromptT5 ethicscommonsensePromptT5 ethicsdeontologyPromptT5 ethicsjusticePromptT5 ethicsvirtuePromptT5

for MODEL in IMDBPromptT5_label laptopPromptT5_label MNLIPromptT5_label QNLIPromptT5_label QQPPromptT5_label restaurantPromptT5_label SST2PromptT5_label snliPromptT5_label tweetevalsentimentPromptT5_label movierationalesPromptT5_label recastnerPromptT5_label ethicsdeontologyPromptT5_label ethicsjusticePromptT5_label MRPCPromptT5_label
'''

BACKBRON="Small"

for MODEL in IMDBPromptT5 laptopPromptT5 MNLIPromptT5 QNLIPromptT5 QQPPromptT5 restaurantPromptT5 SST2PromptT5 snliPromptT5 tweetevalsentimentPromptT5 movierationalesPromptT5 ethicsdeontologyPromptT5 ethicsjusticePromptT5 MRPCPromptT5 multi_newsPromptT5 nq_openPromptT5 samsumPromptT5 squadPromptT5
#for MODEL in IMDBPromptT5Small laptopPromptT5Small restaurantPromptT5Small SST2PromptT5Small tweetevalsentimentPromptT5Small movierationalesPromptT5Small
do
    echo "==========================="
    echo activate_neuronPromptT5Small
    #echo Replace with task_prompt_emb/$MODEL
    echo "==========================="

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron_t5_small.py --config config/activate_neuronPromptT5Small.config \
        --gpu $gpus \
        --replacing_prompt task_prompt_emb/${MODEL}${BACKBRON} \
        --activate_neuron
    #exit
done





###
#cd /data/private/suyusheng/prompt/data/activate_neuron_data
#python3 create_valid.py
#cd /data/private/suyusheng/prompt
###

#python3 activate_neuron_sim_t5.py

'''
for i in 9
do
    python3 activate_neuron_everylayer_sim.py $i
    #echo $i
done
'''




