import argparse
import logging
import random
import numpy as np
import os
import json
import math
import numpy

import torch
import sys


import glob

#####



def EuclideanDistances(task1_emb,task2_emb):
    return float((1/(1+torch.norm(task1_emb-task2_emb, p='fro'))))


def EuclideanDistances_per_token(task1_emb,task2_emb):
    task1_emb = task1_emb.reshape(100,int(task1_emb.shape[-1]/100))
    task2_emb = task2_emb.reshape(100,int(task2_emb.shape[-1]/100))
    sum_euc = 0
    for idx1, v1 in enumerate(task1_emb):
        for idx2, v2 in enumerate(task2_emb):
            #euc = torch.norm(v1-v2, p='fro')
            euc = torch.norm(v1-v2, p=2)
            #print(euc)
            sum_euc += euc
    return float((1/float(1+(float(sum_euc/100)/100))))
    #return torch.norm(task1_emb-task2_emb, p='fro')


#cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
cos = torch.nn.CosineSimilarity(dim=0)
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



#def activate_neurons()
#Neuron_cos = torch.nn.CosineSimilarity(dim=0)
def ActivatedNeurons(activated_1, activated_2, layer, model_name=None):
    activated_1 = activated_1.float()
    activated_2 = activated_2.float()


    if "T5XXL" in model_name:
        if layer == 24:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])

            activated_2 = activated_2[int(layer):int(layer)+3,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])

    elif "Small" in model_name:
        if layer == 6:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:,:,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

            activated_2 = activated_2[int(layer):int(layer)+3,:,:,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])

    elif "Large" in model_name:
        if layer == 24:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:,:,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

            activated_2 = activated_2[int(layer):int(layer)+3,:,:,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
    else:
        if layer == 12:
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





    #print(activated_1.shape)
    #exit()

    #act_1 = set(torch.topk(activated_1, 100).indices.tolist())
    #act_2 = set(torch.topk(activated_2, 100).indices.tolist())
    #sim = len(set.intersection(act_1,act_2))/100


    return float(sim)






root = "task_prompt_emb"



#backbone_model = "Roberta"
#backbone_model = "RobertaLarge"
#backbone_model = "T5"
#backbone_model = "T5Large"
backbone_model = "T5Small"
#backbone_model = "T5XXL"


task_map = {
"IMDBPrompt":0,
"SST2Prompt":0,
"laptopPrompt":0,
"restaurantPrompt":0,
"movierationalesPrompt":0,
"tweetevalsentimentPrompt":0,
"MNLIPrompt":1,
"QNLIPrompt":1,
"snliPrompt":1,
"ethicsdeontologyPrompt":2,
"ethicsjusticePrompt":2,
"QQPPrompt":3,
"MRPCPrompt":3,
"squadPrompt":4,
"nq_openPrompt":4,
"multi_newsPrompt":5,
"samsumPrompt":5}



'''
task_map = {
"IMDBPrompt":0,
"SST2Prompt":0,
"laptopPrompt":0,
"restaurantPrompt":0,
"movierationalesPrompt":0,
"tweetevalsentimentPrompt":0,
"MNLIPrompt":1,
"QNLIPrompt":1,
"snliPrompt":1,
"ethicsdeontologyPrompt":2,
"ethicsjusticePrompt":2,
"QQPPrompt":3,
"MRPCPrompt":3}
'''

'''
task_map = {
"IMDBPrompt":0,
"SST2Prompt":0,
"laptopPrompt":0,
"restaurantPrompt":0,
"movierationalesPrompt":0,
"tweetevalsentimentPrompt":0
}
'''


task_map = {k+backbone_model:v for k,v in task_map.items() }



#for metric in ["CosineSimilarity", "CosineSimilarity_per_token", "EuclideanDistances", "EuclideanDistances_per_token", "ActivatedNeurons_0", "ActivatedNeurons_3", "ActivatedNeurons_6", "ActivatedNeurons_9", "ActivatedNeurons_12","ActivatedNeurons_15", "ActivatedNeurons_18", "ActivatedNeurons_21", "ActivatedNeurons_24"]:
#for metric in ["CosineSimilarity", "CosineSimilarity_per_token","ActivatedNeurons_0", "ActivatedNeurons_3", "ActivatedNeurons_6", "ActivatedNeurons_9", "ActivatedNeurons_12"]:
for metric in ["ActivatedNeurons_0", "ActivatedNeurons_3", "ActivatedNeurons_6", "ActivatedNeurons_9", "ActivatedNeurons_12"]:
#for metric in ["ActivatedNeurons_0", "ActivatedNeurons_3", "ActivatedNeurons_6", "ActivatedNeurons_9", "ActivatedNeurons_12", "ActivatedNeurons_15", "ActivatedNeurons_18", "ActivatedNeurons_21", "ActivatedNeurons_24"]:
#for metric in ["ActivatedNeurons_24"]:
#for metric in ["CosineSimilarity", "CosineSimilarity_per_token","EuclideanDistances", "EuclideanDistances_per_token","ActivatedNeurons_0", "ActivatedNeurons_3", "ActivatedNeurons_6"]:
#for metric in ["ActivatedNeurons_0", "ActivatedNeurons_3", "ActivatedNeurons_6"]:

    print()
    print("=========")
    print(metric)
    print("=========")

    #print(end="\t")
    for name, id in task_map.items():
        #print(name, end='\t')
        name = name.replace("PromptRoberta","").replace("ethics","").replace("recast","")
        if len(name)>5:
            name = name[:5]
        print(name, end="\t")
    print()




    for task_1, id_1 in task_map.items():
        name_1 = task_1
        task_ten_1 = torch.load(root+"/"+name_1+"/task_prompt", map_location=lambda storage, loc: storage)
        task_ten_1 = task_ten_1.reshape(task_ten_1.shape[0]*task_ten_1.shape[1])

        if "T5XXL" in name_1:
            task_ten_1_neurons = torch.load("task_activated_neuron"+"/"+name_1+"/task_activated_neuron", map_location=lambda storage, loc: storage)
            task_ten_1_neurons = task_ten_1_neurons.reshape(24,10240)
        elif "T5Small" in name_1:
            task_ten_1_neurons = torch.load("task_activated_neuron"+"/"+name_1+"/task_activated_neuron", map_location=lambda storage, loc: storage)
            task_ten_1_neurons = task_ten_1_neurons[:,0:1,:,:]
        elif "T5" in name_1:
            task_ten_1_neurons = torch.load("task_activated_neuron"+"/"+name_1+"decoder/task_activated_neuron", map_location=lambda storage, loc: storage)
            task_ten_1_neurons = task_ten_1_neurons[:,0:1,:,:]
        else:
            task_ten_1_neurons = torch.load("task_activated_neuron"+"/"+name_1+"/task_activated_neuron", map_location=lambda storage, loc: storage)
            #print(task_ten_1_neurons.shape)
            #exit()

        name_1 = name_1.replace("PromptRoberta","").replace("ethics","").replace("recast","")
        if len(name_1)>5:
            name_1 = name_1[:5]

        #print(name_1, end="\t")


        print("[", end=' ')
        for task_2, id_2 in task_map.items():
            name_2 = task_2
            task_ten_2 = torch.load(root+"/"+name_2+"/task_prompt", map_location=lambda storage, loc: storage)
            task_ten_2 = task_ten_2.reshape(task_ten_2.shape[0]*task_ten_2.shape[1])

            if "T5XXL" in name_2:
                task_ten_2_neurons = torch.load("task_activated_neuron"+"/"+name_2+"/task_activated_neuron", map_location=lambda storage, loc: storage)
                task_ten_2_neurons = task_ten_2_neurons.reshape(24,10240)
            elif "T5Small" in name_2:
                task_ten_2_neurons = torch.load("task_activated_neuron"+"/"+name_2+"/task_activated_neuron", map_location=lambda storage, loc: storage)
                task_ten_2_neurons = task_ten_2_neurons[:,0:1,:,:]
            elif "T5" in name_2:
                task_ten_2_neurons = torch.load("task_activated_neuron"+"/"+name_2+"decoder/task_activated_neuron", map_location=lambda storage, loc: storage)
                task_ten_2_neurons = task_ten_2_neurons[:,0:1,:,:]
            else:
                task_ten_2_neurons = torch.load("task_activated_neuron"+"/"+name_2+"/task_activated_neuron", map_location=lambda storage, loc: storage)

            #["CosineSimilarity", "CosineSimilarity_per_token", "EuclideanDistances", "EuclideanDistances_per_token", "ActivatedNeurons_3", "ActivatedNeurons_6", "ActivatedNeurons_9", "ActivatedNeurons_12"]

            if metric == "CosineSimilarity":
                sim = CosineSimilarity(task_ten_1,task_ten_2)
                print("{:.4f},".format(float(sim)), end='\t')
            elif metric == "CosineSimilarity_per_token":
                sim = CosineSimilarity_per_token(task_ten_1,task_ten_2)
                print("{:.4f},".format(float(sim)), end='\t')
            elif metric == "EuclideanDistances":
                sim = EuclideanDistances(task_ten_1,task_ten_2)
                print("{:.4f},".format(float(sim)), end='\t')
            elif metric == "EuclideanDistances_per_token":
                sim = EuclideanDistances_per_token(task_ten_1,task_ten_2)
                print("{:.4f},".format(float(sim)), end='\t')


            elif metric == "ActivatedNeurons_0":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 0, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_3":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 3, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_6":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 6, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_9":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 9, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_12":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 12, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_15":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 15, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_18":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 18, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_21":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 21, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            elif metric == "ActivatedNeurons_24":
                sim = ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 24, backbone_model)
                print("{:.2f},".format(float(sim)), end='\t')
            else:
                continue
        #print("]")



        print()



