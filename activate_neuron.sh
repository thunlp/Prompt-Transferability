gpus=4



'''
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta WNLIPromptRoberta anliPromptRoberta snliPromptRoberta tweetevalsentimentPromptRoberta movierationalesPromptRoberta recastfactualityPromptRoberta recastpunsPromptRoberta recastverbnetPromptRoberta recastverbcornerPromptRoberta recastnerPromptRoberta recastsentimentPromptRoberta recastmegaveridicalityPromptRoberta ethicscommonsensePromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta ethicsvirtuePromptRoberta

#for MODEL in recastfactualityPromptRoberta recastpunsPromptRoberta recastverbnetPromptRoberta recastverbcornerPromptRoberta recastnerPromptRoberta recastsentimentPromptRoberta recastmegaveridicalityPromptRoberta ethicscommonsensePromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta ethicsvirtuePromptRoberta
do
    echo "==========================="
    echo activate_neuronPromptRoberta
    echo Replace with task_prompt_emb/$MODEL
    echo "==========================="

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py --config config/activate_neuronPromptRoberta.config \
        --gpu $gpus \
        --replacing_prompt task_prompt_emb/$MODEL \
        --activate_neuron
done
'''



###
#cd /data/private/suyusheng/prompt/data/activate_neuron_data
#python3 create_valid.py
#cd /data/private/suyusheng/prompt
###

#python3 activate_neuron_sim.py


for i in {0..11}
do
    python3 activate_neuron_everylayer_sim.py $i
done




'''
for i in {1..3}
do
    for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta WNLIPromptRoberta anliPromptRoberta snliPromptRoberta tweetevalsentimentPromptRoberta movierationalesPromptRoberta
    do
        echo "==========================="
        echo activate_neuronPromptRoberta
        echo Replace with task_prompt_emb/$MODEL
        echo "==========================="

        #Eval mlm
        CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py --config config/activate_neuronPromptRoberta.config \
            --gpu $gpus \
            --replacing_prompt task_prompt_emb/$MODEL \
            --activate_neuron
    done

    ###
    cd /data/private/suyusheng/prompt/data/activate_neuron_data
    python3 create_valid.py
    cd /data/private/suyusheng/prompt
    ###

    python3 activate_neuron_sim.py $i

done
'''



#cd task_activated_neuron/
#mv *PromptRoberta 12layer_1prompt


#--replacing_prompt task_prompt_emb/$MODEL \
#--replacing_prompt random \
#--checkpoint model/$MODEL \


'''
CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py \
    --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta
'''

#python3 activate_neuron_sim.py


