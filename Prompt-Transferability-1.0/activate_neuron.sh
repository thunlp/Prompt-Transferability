gpus=7

######################################
#######Activate Neurons Roberta#######
######################################

BACKBONE_MODEL="Roberta"
#BACKBONE_MODEL="RobertaLarge"

for MODEL in IMDBPrompt${BACKBONE_MODEL} laptopPrompt${BACKBONE_MODEL} MNLIPrompt${BACKBONE_MODEL} QNLIPrompt${BACKBONE_MODEL} QQPPrompt${BACKBONE_MODEL} restaurantPrompt${BACKBONE_MODEL} SST2Prompt${BACKBONE_MODEL} snliPrompt${BACKBONE_MODEL} tweetevalsentimentPrompt${BACKBONE_MODEL} movierationalesPrompt${BACKBONE_MODEL} recastnerPrompt${BACKBONE_MODEL} ethicsdeontologyPrompt${BACKBONE_MODEL} ethicsjusticePrompt${BACKBONE_MODEL} MRPCPrompt${BACKBONE_MODEL}
do
    echo "==========================="
    echo Activate_neuronPrompt${BACKBONE_MODEL}
    echo Stimulate neurons with task_prompt_emb/$MODEL
    echo "==========================="

    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron_${BACKBONE_MODEL}.py --config config/activate_neuronPrompt${BACKBONE_MODEL}.config \
        --gpu $gpus \
        --replacing_prompt task_prompt_emb/$MODEL \
        --activate_neuron
done




######################################
#########Activate Neurons T5##########
######################################

BACKBONE_MODEL="T5"
#BACKBONE_MODEL="T5SMALL"
#BACKBONE_MODEL="T5Large"

for MODEL in IMDBPrompt${BACKBONE_MODEL} laptopPrompt${BACKBONE_MODEL} MNLIPrompt${BACKBONE_MODEL} QNLIPrompt${BACKBONE_MODEL} QQPPrompt${BACKBONE_MODEL} restaurantPrompt${BACKBONE_MODEL} SST2Prompt${BACKBONE_MODEL} snliPrompt${BACKBONE_MODEL} tweetevalsentimentPrompt${BACKBONE_MODEL} movierationalesPrompt${BACKBONE_MODEL} ethicsdeontologyPrompt${BACKBONE_MODEL} ethicsjusticePrompt${BACKBONE_MODEL} MRPCPrompt${BACKBONE_MODEL} multi_newsPrompt${BACKBONE_MODEL} nq_openPrompt${BACKBONE_MODEL} samsumPrompt${BACKBONE_MODEL} squadPrompt${BACKBONE_MODEL}
do
    echo "==========================="
    echo Activate_neuronPrompt${BACKBONE_MODEL}
    echo Stimulate neurons with task_prompt_emb/$MODEL
    echo "==========================="

    CUDA_VISIBLE_DEVICES=$gpus python3 activate_neuron_${BACKBONE_MODEL}.py --config config/activate_neuronPrompt${BACKBONE_MODEL}.config \
        --gpu $gpus \
        --replacing_prompt task_prompt_emb/$MODEL \
        --activate_neuron
done



'''
for i in 9
do
    python3 activate_neuron_everylayer_sim.py $i
    #echo $i
done
'''




