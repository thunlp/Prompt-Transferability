gpus=2

#STSBPromptRoberta

#for MODEL in WNLIPromptRoberta
#CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta WNLIPromptRoberta
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



#--replacing_prompt task_prompt_emb/$MODEL \
#--replacing_prompt random \
#--checkpoint model/$MODEL \


'''
CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py \
    --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta
'''

python3 activate_neuron_sim.py


