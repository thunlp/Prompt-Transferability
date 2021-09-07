gpus=2

#STSBPromptRoberta

#CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta MRPCPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta RTEPromptRoberta SST2PromptRoberta WNLIPromptRoberta
do
    echo "==========================="
    echo config/tweetevalsentimentPromptRoberta.config
    echo Replace with task_prompt_emb/$MODEL
    echo "==========================="

    #Eval mlm
    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py --config config/tweetevalsentimentPromptRoberta.config \
        --gpu $gpus \
        --replacing_prompt task_prompt_emb/$MODEL
        #--checkpoint model/$MODEL \
done




'''
CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron.py \
    --config config/restaurantPromptRoberta.config \
    --gpu $gpus \
    --checkpoint model/restaurantPromptRoberta
'''


