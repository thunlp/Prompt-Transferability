gpus=7

FROM="T5"
# 22_model_cross_0.7906.pkl  75_model_cross_0.7719.pkl  98_model_cross_0.7469.pkl 23_model_cross_0.7797.pkl  78_model_cross_0.7562.pkl
#MODEL_PROJ="model/crossPromptRoberta_laptop_100_t5_to_roberta/22_model_cross_0.7906.pkl"
#MODEL_PROJ="model/crossPromptRoberta_laptop_100_t5_to_roberta/23_model_cross_0.7797.pkl"
#MODEL_PROJ="model/crossPromptRoberta_laptop_100_t5_to_roberta/75_model_cross_0.7719.pkl"
#MODEL_PROJ="model/crossPromptRoberta_laptop_100_t5_to_roberta/21_model_cross_0.7734.pkl"
#45
#MODEL_PROJ="model/crossPromptRoberta_laptop_100_t5_to_roberta/23_model_cross_0.7531.pkl"
#MODEL_PROJ="model/crossPromptRoberta_laptop_100_t5_to_roberta/38_model_cross_0.7484.pkl"
#438

#MODEL_PROJ="model/crossPromptRoberta_MNLI_100_t5_to_roberta/5_model_cross_0.7631.pkl"
#MODEL_PROJ="model/crossPromptRoberta_MNLI_100_t5_to_roberta/3_model_cross_0.714.pkl"
#MODEL_PROJ="model/crossPromptRoberta_MNLI_100_t5_to_roberta/7_model_cross_0.7987.pkl"
MODEL_PROJ="model/crossPromptRoberta_MNLI_100_t5_to_roberta/19_model_cross_0.8227.pkl"

#for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta SST2PromptRoberta snliPromptRoberta tweetevalsentimentPromptRoberta movierationalesPromptRoberta recastnerPromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta MRPCPromptRoberta
for MODEL in MNLIPrompt snliPrompt QNLIPrompt
#for MODEL in laptopPrompt IMDBPrompt restaurantPrompt SST2Prompt tweetevalsentimentPrompt movierationalesPrompt
do
CUDA_VISIBLE_DEVICES=$gpus python3 caculate_projected_save_proj_prompt_and_activated_neurons.py --config config/activate_neuronPromptRoberta.config \
    --gpu $gpus \
    --replacing_prompt task_prompt_emb/$MODEL${FROM} \
    --projector ${MODEL_PROJ} \
    --model_transfer_projector \
    --activate_neuron
done



'''
for MODEL in IMDBPromptRoberta laptopPromptRoberta MNLIPromptRoberta QNLIPromptRoberta QQPPromptRoberta restaurantPromptRoberta SST2PromptRoberta snliPromptRoberta tweetevalsentimentPromptRoberta movierationalesPromptRoberta recastnerPromptRoberta ethicsdeontologyPromptRoberta ethicsjusticePromptRoberta MRPCPromptRoberta
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







