import torch
import os


def EuclideanDistances(task1_emb,task2_emb):
    return float((1/(1+torch.norm(task1_emb-task2_emb, p='fro'))))


def EuclideanDistances_per_token(task1_emb,task2_emb):
    task1_emb = task1_emb.reshape(100,int(task1_emb.shape[-1]/100))
    task2_emb = task2_emb.reshape(100,int(task2_emb.shape[-1]/100))
    sum_euc = 0
    for idx1, v1 in enumerate(task1_emb):
        #print(idx1)
        #print(v1)
        #print(v1.shape)
        #exit()
        for idx2, v2 in enumerate(task2_emb):
            #euc = torch.norm(v1-v2, p='fro')
            euc = torch.norm(v1-v2, p=2)
            #print(euc)
            sum_euc += euc
    return float((1/float(1+(float(sum_euc/100)/100))))
    #return torch.norm(task1_emb-task2_emb, p='fro')


cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    #return cos(task1_emb,task2_emb).sum()
    return float(cos(task1_emb,task2_emb))

def CosineSimilarity_per_token(task1_emb,task2_emb):
    #print(int(task1_emb.shape[-1]/100))
    task1_emb = task1_emb.reshape(100,int(task1_emb.shape[-1]/100))
    task2_emb = task2_emb.reshape(100,int(task2_emb.shape[-1]/100))
    sum_c = 0
    #return cos(task1_emb,task2_emb).sum()
    for idx1, v1 in enumerate(task1_emb):
        for idx2, v2 in enumerate(task2_emb):
            c = cos(v1,v2)
            sum_c += c
    return float(float(sum_c/100)/100)


def ActivatedNeurons(activated_1, activated_2, layer, backbone_model=None):

    activated_1 = activated_1.float()
    activated_2 = activated_2.float()

    if backbone_model == "T5XXL":
        if layer == 24:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])

            activated_2 = activated_2[int(layer):int(layer)+3,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])

    elif backbone_model == "RobertaLarge":
        if layer ==24:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:,:,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

            activated_2 = activated_2[int(layer):int(layer)+3,:,:,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])


    else:
        if layer ==12:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:,:,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

            activated_2 = activated_2[int(layer):int(layer)+3,:,:,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])


    activated_1[activated_1>0] = float(1)
    activated_1[activated_1<0] = float(0)

    activated_2[activated_2>0] = float(1)
    activated_2[activated_2<0] = float(0)

    sim = cos(activated_1, activated_2)

    return float(sim)




root_dir1 = "task_prompt_emb/RobertaLarge_from_chimin/"
#root_dir1 = "task_prompt_emb/T5XXL_from_chimin/"
root_dir2 = "task_prompt_emb/"

root_dir1_act = "task_activated_neuron/RobertaLarge_from_chimin/"
#root_dir1_act = "task_activated_neuron/old_PromptT5XXL_activated_neurons/"
root_dir2_act = "task_activated_neuron/"

#root_prompts = os.listdir("task_prompt_emb/RobertaLarge_from_chimin")
root_prompts = os.listdir("task_prompt_emb/RobertaLarge_from_chimin")
#root_prompts = os.listdir("task_prompt_emb/T5XXL_from_chimin")


same_task_list_cos = list()
same_task_list_cos_per_token = list()
same_task_list_euc = list()
same_task_list_euc_per_token = list()
same_task_list_neurons_0 = list()
same_task_list_neurons_3 = list()
same_task_list_neurons_6 = list()
same_task_list_neurons_9 = list()
same_task_list_neurons_12 = list()
same_task_list_neurons_15 = list()
same_task_list_neurons_18 = list()
same_task_list_neurons_21 = list()
same_task_list_neurons_24 = list()

#backbone_model = "RobertaLarge"
backbone_model = "T5XXL"
#backbone_model = "T5Small"

for prompt in root_prompts:
    if "random" in prompt or ".py" in prompt or "snli" in prompt:
        continue

    task_ten_1 = torch.load(root_dir1+prompt+"/task_prompt", map_location=lambda storage, loc: storage)
    task_ten_1 = task_ten_1.reshape(task_ten_1.shape[0]*task_ten_1.shape[1])
    task_ten_2 = torch.load(root_dir2+prompt+"/task_prompt", map_location=lambda storage, loc: storage)
    task_ten_2 = task_ten_2.reshape(task_ten_2.shape[0]*task_ten_2.shape[1])

    if backbone_model == "T5XXL":
        task_ten_1_neurons = torch.load(root_dir1_act+prompt+"/task_activated_neuron", map_location=lambda storage, loc: storage)
        task_ten_1_neurons = task_ten_1_neurons.reshape(24,10240)
        task_ten_2_neurons = torch.load(root_dir2_act+prompt+"/task_activated_neuron", map_location=lambda storage, loc: storage)
        task_ten_2_neurons = task_ten_2_neurons.reshape(24,10240)
    else:
        task_ten_1_neurons = torch.load(root_dir1_act+prompt+"/task_activated_neuron", map_location=lambda storage, loc: storage)
        task_ten_1_neurons = task_ten_1_neurons[:,0:1,:,:]
        task_ten_2_neurons = torch.load(root_dir2_act+prompt+"/task_activated_neuron", map_location=lambda storage, loc: storage)
        task_ten_2_neurons = task_ten_2_neurons[:,0:1,:,:]


    same_task_list_cos.append(CosineSimilarity(task_ten_1,task_ten_2))
    same_task_list_cos_per_token.append(CosineSimilarity_per_token(task_ten_1,task_ten_2))
    same_task_list_euc.append(EuclideanDistances(task_ten_1,task_ten_2))
    same_task_list_euc_per_token.append(EuclideanDistances_per_token(task_ten_1,task_ten_2))
    same_task_list_neurons_0.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 0, backbone_model))
    same_task_list_neurons_3.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 3, backbone_model))
    same_task_list_neurons_6.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 6, backbone_model))
    same_task_list_neurons_9.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 9, backbone_model))
    same_task_list_neurons_12.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 12, backbone_model))
    same_task_list_neurons_15.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 15, backbone_model))
    same_task_list_neurons_18.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 18, backbone_model))
    same_task_list_neurons_21.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 21, backbone_model))
    same_task_list_neurons_24.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 24, backbone_model))


print("same_task_list_euc:", sum(same_task_list_euc)/len(same_task_list_euc))
print("same_task_list_euc_per_token:", sum(same_task_list_euc_per_token)/len(same_task_list_euc_per_token))
print("same_task_list_cos:", sum(same_task_list_cos)/len(same_task_list_cos))
print("same_task_list_cos_per_token:", sum(same_task_list_cos_per_token)/len(same_task_list_cos_per_token))
print("same_task_list_neurons_0:", sum(same_task_list_neurons_0)/len(same_task_list_neurons_0))
print("same_task_list_neurons_3:", sum(same_task_list_neurons_3)/len(same_task_list_neurons_3))
print("same_task_list_neurons_6:", sum(same_task_list_neurons_6)/len(same_task_list_neurons_6))
print("same_task_list_neurons_9:", sum(same_task_list_neurons_9)/len(same_task_list_neurons_9))
print("same_task_list_neurons_12:", sum(same_task_list_neurons_12)/len(same_task_list_neurons_12))
if len(same_task_list_neurons_15)!=0:
    print("same_task_list_neurons_15:", sum(same_task_list_neurons_15)/len(same_task_list_neurons_15))
    print("same_task_list_neurons_18:", sum(same_task_list_neurons_18)/len(same_task_list_neurons_18))
    print("same_task_list_neurons_21:", sum(same_task_list_neurons_21)/len(same_task_list_neurons_21))
    print("same_task_list_neurons_24:", sum(same_task_list_neurons_24)/len(same_task_list_neurons_24))

print("================")
