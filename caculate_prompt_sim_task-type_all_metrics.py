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


#from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
#from openTSNE import initialization
#from openTSNE.callbacks import ErrorLogger
#from examples import utils
#from openTSNE_.examples import utils_
#import utils
#import numpy as np
#import matplotlib.pyplot as plt

#from tsnecuda import TSNE
#from os import listdir
#from os.path import isfile, join
import glob

#####


#task_ten={0:sst2_ten,1:rte_ten,2:re_ten,3:MNLI_ten,4:MRPC_ten,5:QNLI_ten,6:QQP_ten,7:WNLI_ten,8:STSB_ten}

#task_map={0:"sst2_15",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB"}



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



#def activate_neurons()
#Neuron_cos = torch.nn.CosineSimilarity(dim=0)
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

    elif "Small" in backbone_model:
        if layer ==6:
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
            '''
            activated_1 = activated_1[int(layer):int(layer)+6,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])

            activated_2 = activated_2[int(layer):int(layer)+6,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])
            '''


    activated_1[activated_1>0] = float(1)
    activated_1[activated_1<0] = float(0)

    activated_2[activated_2>0] = float(1)
    activated_2[activated_2<0] = float(0)

    sim = cos(activated_1, activated_2)

    return float(sim)






root = "task_prompt_emb"



#backbone_model = "Roberta"
#backbone_model = "RobertaLarge"
#backbone_model = "T5"
#backbone_model = "T5XXL"
backbone_model = "T5Small"

task_map = {"IMDBPrompt":0, "laptopPrompt":0, "MNLIPrompt":1, "QNLIPrompt":1, "QQPPrompt":3, "restaurantPrompt":0, "SST2Prompt":0, "snliPrompt":1, "tweetevalsentimentPrompt":0, "movierationalesPrompt":0, "ethicsdeontologyPrompt":2, "ethicsjusticePrompt":2, "MRPCPrompt":3, "squadPrompt":4, "nq_openPrompt":4, "multi_newsPrompt":5, "samsumPrompt":5}
#task_map = {"IMDBPrompt":0, "laptopPrompt":0, "MNLIPrompt":1, "QNLIPrompt":1, "QQPPrompt":3, "restaurantPrompt":0, "SST2Prompt":0, "snliPrompt":1, "tweetevalsentimentPrompt":0, "movierationalesPrompt":0, "ethicsdeontologyPrompt":2, "ethicsjusticePrompt":2, "MRPCPrompt":3}

task_map = {k+backbone_model:v for k,v in task_map.items()}
#print(task_map)
#exit()


#task_map = {0:"IMDBPromptRoberta",1:"SST2PromptRoberta",2:"laptopPromptRoberta",3:"restaurantPromptRoberta",4:"movierationalesPromptRoberta",5:"tweetevalsentimentPromptRoberta",6:"MNLIPromptRoberta",7:"QNLIPromptRoberta",8:"snliPromptRoberta",9:"ethicsdeontologyPromptRoberta",10:"ethicsjusticePromptRoberta",11:"QQPPromptRoberta",12:"MRPCPromptRoberta"}




#sys.stdout = open("task_cos_distance.txt", 'w')
#sys.stdout = open("task_ecd_distance.txt", 'w')


'''
print(end="\t")
for name, id in task_map.items():
    #print(name, end='\t')
    name = name.replace("PromptRoberta","").replace("ethics","").replace("recast","")
    if len(name)>5:
        name = name[:5]
    print(name, end="\t")
print()
'''


l=0




same_type_list_cos = list()
same_type_list_cos_per_token = list()
same_type_list_euc = list()
same_type_list_euc_per_token = list()
same_type_list_neurons_0 = list()
same_type_list_neurons_3 = list()
same_type_list_neurons_6 = list()
same_type_list_neurons_9 = list()
same_type_list_neurons_12 = list()
same_type_list_neurons_15 = list()
same_type_list_neurons_18 = list()
same_type_list_neurons_21 = list()
same_type_list_neurons_24 = list()


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

different_type_list_cos = list()
different_type_list_cos_per_token = list()
different_type_list_euc = list()
different_type_list_euc_per_token = list()
different_type_list_neurons_0 = list()
different_type_list_neurons_3 = list()
different_type_list_neurons_6 = list()
different_type_list_neurons_9 = list()
different_type_list_neurons_12 = list()
different_type_list_neurons_15 = list()
different_type_list_neurons_18 = list()
different_type_list_neurons_21 = list()
different_type_list_neurons_24 = list()


different_task_list_cos = list()
different_task_list_cos_per_token = list()
different_task_list_euc = list()
different_task_list_euc_per_token = list()
different_task_list_neurons_0 = list()
different_task_list_neurons_3 = list()
different_task_list_neurons_6 = list()
different_task_list_neurons_9 = list()
different_task_list_neurons_12 = list()
different_task_list_neurons_15 = list()
different_task_list_neurons_18 = list()
different_task_list_neurons_21 = list()
different_task_list_neurons_24 = list()


for task_1, id_1 in task_map.items():
    #if id_1 not in show_in_list:
    #    continue
    #cos_dict=dict()
    #euc_dict=dict()
    '''
    if task_1 == "rest":
        name_1 = "restaurant"
    elif task_1 == "movie":
        name_1 = "movierationales"
    elif task_1 == "tweet":
        name_1 = "tweetevalsentiment"
    '''
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

    name_1 = name_1.replace("PromptRoberta","").replace("ethics","").replace("recast","")
    if len(name_1)>5:
        name_1 = name_1[:5]
    #print(name_1, end="\t")
    for task_2, id_2 in task_map.items():
        #if id_2 not in show_in_list:
        #    continue
        #if id_1 == id_2:
        #    continue
        #else:
        #similiarty:
        #cos:
        '''
        if task_2 == "rest":
            name_2 = "restaurant"
        elif task_2 == "movie":
            name_2 = "movierationales"
        elif task_2 == "tweet":
            name_2 = "tweetevalsentiment"
        else:
        '''
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


        #ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons)
        #exit()

        '''
        #same task
        if task_1 == task_2 and id_1 == id_2:
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

        # different task
        if task_1 != task_2:
            #print("00",id_1,id_2)
            different_task_list_cos.append(CosineSimilarity(task_ten_1,task_ten_2))
            different_task_list_cos_per_token.append(CosineSimilarity_per_token(task_ten_1,task_ten_2))
            different_task_list_euc.append(EuclideanDistances(task_ten_1,task_ten_2))
            different_task_list_euc_per_token.append(EuclideanDistances_per_token(task_ten_1,task_ten_2))
            different_task_list_neurons_0.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 0, backbone_model))
            different_task_list_neurons_3.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 3, backbone_model))
            different_task_list_neurons_6.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 6, backbone_model))
            different_task_list_neurons_9.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 9, backbone_model))
            different_task_list_neurons_12.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 12, backbone_model))
            different_task_list_neurons_15.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 15, backbone_model))
            different_task_list_neurons_18.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 18, backbone_model))
            different_task_list_neurons_21.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 21, backbone_model))
            different_task_list_neurons_24.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 24, backbone_model))
        '''

        #same task type
        if task_1 != task_2 and id_1 == id_2:
            same_type_list_cos.append(CosineSimilarity(task_ten_1,task_ten_2))
            same_type_list_cos_per_token.append(CosineSimilarity_per_token(task_ten_1,task_ten_2))
            same_type_list_euc.append(EuclideanDistances(task_ten_1,task_ten_2))
            same_type_list_euc_per_token.append(EuclideanDistances_per_token(task_ten_1,task_ten_2))
            same_type_list_neurons_0.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 0, backbone_model))
            same_type_list_neurons_3.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 3, backbone_model))
            same_type_list_neurons_6.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 6, backbone_model))
            same_type_list_neurons_9.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 9, backbone_model))
            same_type_list_neurons_12.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 12, backbone_model))
            same_type_list_neurons_15.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 15, backbone_model))
            same_type_list_neurons_18.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 18, backbone_model))
            same_type_list_neurons_21.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 21, backbone_model))
            same_type_list_neurons_24.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 24, backbone_model))

        '''
        # different task type
        if task_1 != task_2 and id_1 != id_2:
            #print("11",id_1,id_2)
            different_type_list_cos.append(CosineSimilarity(task_ten_1,task_ten_2))
            different_type_list_cos_per_token.append(CosineSimilarity_per_token(task_ten_1,task_ten_2))
            different_type_list_euc.append(EuclideanDistances(task_ten_1,task_ten_2))
            different_type_list_euc_per_token.append(EuclideanDistances_per_token(task_ten_1,task_ten_2))
            different_type_list_neurons_0.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 0, backbone_model))
            different_type_list_neurons_3.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 3, backbone_model))
            different_type_list_neurons_6.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 6, backbone_model))
            different_type_list_neurons_9.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 9, backbone_model))
            different_type_list_neurons_12.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 12, backbone_model))
            different_type_list_neurons_15.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 15, backbone_model))
            different_type_list_neurons_18.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 18, backbone_model))
            different_type_list_neurons_21.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 21, backbone_model))
            different_type_list_neurons_24.append(ActivatedNeurons(task_ten_1_neurons, task_ten_2_neurons, 24, backbone_model))
        '''

        #break


        #cos_dict[task_2]=float(CosineSimilarity(task_ten_1,task_ten_2))
        #sim=float(CosineSimilarity(task_ten_1,task_ten_2))
        #print(sim)
        #exit()

        #endcli
        #euc_dict[task_2]=float(EuclideanDistances(task_ten_1,task_ten_2))
        #sim=float(EuclideanDistances(task_ten_1,task_ten_2))
        #sim=float((float(EuclideanDistances(task_ten_1,task_ten_2))+1))
        #sim=float(CosineSimilarity_per_token(task_ten_1,task_ten_2))

        #sim=float(CosineSimilarity_avg(task_ten_1,task_ten_2))
        #sim=float(EuclideanDistances_avg(task_ten_1,task_ten_2))

        #sim=float(EuclideanDistances_per_token(task_ten_1,task_ten_2))
        #task_ten_1 = task_ten_1.reshape(1,76800)
        #print(task_ten_1.shape)
        #exit()
        #sim=float(Euclidean(task_ten_1,task_ten_2))


        #print(sim, end='\t')
        #print("{:.2f},".format(float(sim)), end='\t')
        #print("{:.0f}".format(float(sim)), end='\t')
        #print("{:.5f}".format(float(sim)),",", end='\t')
        #print("{:.4f}".format(float(sim)), end=' ')


        #if task_1 != task_2:
        #if task_1 == task_2:
            #print(sim)
        #    l+=sim


    #print()

#print("All diferent tasks", l/(13*12))
#print("same tasks", l/(13*12))
#print("Same Task:", same_task)



print("result")

print("same_type_list_euc:", sum(same_type_list_euc)/len(same_type_list_euc))
print("same_type_list_euc_per_token:", sum(same_type_list_euc_per_token)/len(same_type_list_euc_per_token))
print("same_type_list_cos:", sum(same_type_list_cos)/len(same_type_list_cos))
print("same_type_list_cos_per_token:", sum(same_type_list_cos_per_token)/len(same_type_list_cos_per_token))
print("same_type_list_neurons_0:", sum(same_type_list_neurons_0)/len(same_type_list_neurons_0))
print("same_type_list_neurons_3:", sum(same_type_list_neurons_3)/len(same_type_list_neurons_3))
print("same_type_list_neurons_6:", sum(same_type_list_neurons_6)/len(same_type_list_neurons_6))
if len(same_type_list_neurons_9)!=0:
    print("same_type_list_neurons_9:", sum(same_type_list_neurons_9)/len(same_type_list_neurons_9))
    print("same_type_list_neurons_12:", sum(same_type_list_neurons_12)/len(same_type_list_neurons_12))
if len(same_type_list_neurons_15)!=0:
    print("same_type_list_neurons_15:", sum(same_type_list_neurons_15)/len(same_type_list_neurons_15))
    print("same_type_list_neurons_18:", sum(same_type_list_neurons_18)/len(same_type_list_neurons_18))
    print("same_type_list_neurons_21:", sum(same_type_list_neurons_21)/len(same_type_list_neurons_21))
    print("same_type_list_neurons_24:", sum(same_type_list_neurons_24)/len(same_type_list_neurons_24))

print("================")

print("same_task_list_euc:", sum(same_task_list_euc)/len(same_task_list_euc))
print("same_task_list_euc_per_token:", sum(same_task_list_euc_per_token)/len(same_task_list_euc_per_token))
print("same_task_list_cos:", sum(same_task_list_cos)/len(same_task_list_cos))
print("same_task_list_cos_per_token:", sum(same_task_list_cos_per_token)/len(same_task_list_cos_per_token))
print("same_task_list_neurons_0:", sum(same_task_list_neurons_0)/len(same_task_list_neurons_0))
print("same_task_list_neurons_3:", sum(same_task_list_neurons_3)/len(same_task_list_neurons_3))
print("same_task_list_neurons_6:", sum(same_task_list_neurons_6)/len(same_task_list_neurons_6))
if len(same_task_list_neurons_9)!=0:
    print("same_task_list_neurons_9:", sum(same_task_list_neurons_9)/len(same_task_list_neurons_9))
    print("same_task_list_neurons_12:", sum(same_task_list_neurons_12)/len(same_task_list_neurons_12))
if len(same_task_list_neurons_15)!=0:
    print("same_task_list_neurons_15:", sum(same_task_list_neurons_15)/len(same_task_list_neurons_15))
    print("same_task_list_neurons_18:", sum(same_task_list_neurons_18)/len(same_task_list_neurons_18))
    print("same_task_list_neurons_21:", sum(same_task_list_neurons_21)/len(same_task_list_neurons_21))
    print("same_task_list_neurons_24:", sum(same_task_list_neurons_24)/len(same_task_list_neurons_24))

print("================")

print("different_type_list_euc:", sum(different_type_list_euc)/len(different_type_list_euc))
print("different_type_list_euc_per_token:", sum(different_type_list_euc_per_token)/len(different_type_list_euc_per_token))
print("different_type_list_cos:", sum(different_type_list_cos)/len(different_type_list_cos))
print("different_type_list_cos_per_token:", sum(different_type_list_cos_per_token)/len(different_type_list_cos_per_token))
print("different_type_list_neurons_0:", sum(different_type_list_neurons_0)/len(different_type_list_neurons_0))
print("different_type_list_neurons_3:", sum(different_type_list_neurons_3)/len(different_type_list_neurons_3))
print("different_type_list_neurons_6:", sum(different_type_list_neurons_6)/len(different_type_list_neurons_6))
if len(different_type_list_neurons_9)!=0:
    print("different_type_list_neurons_9:", sum(different_type_list_neurons_9)/len(different_type_list_neurons_9))
    print("different_type_list_neurons_12:", sum(different_type_list_neurons_12)/len(different_type_list_neurons_12))
if len(different_type_list_neurons_15)!=0:
    print("different_type_list_neurons_15:", sum(different_type_list_neurons_15)/len(different_type_list_neurons_15))
    print("different_type_list_neurons_18:", sum(different_type_list_neurons_18)/len(different_type_list_neurons_18))
    print("different_type_list_neurons_21:", sum(different_type_list_neurons_21)/len(different_type_list_neurons_21))
    print("different_type_list_neurons_24:", sum(different_type_list_neurons_24)/len(different_type_list_neurons_24))

print("================")

print("different_task_list_euc", sum(different_task_list_euc)/len(different_task_list_euc))
print("different_task_list_euc_per_token:", sum(different_task_list_euc_per_token)/len(different_task_list_euc_per_token))
print("different_task_list_cos:", sum(different_task_list_cos)/len(different_task_list_cos))
print("different_task_list_cos_per_token:", sum(different_task_list_cos_per_token)/len(different_task_list_cos_per_token))
print("different_task_list_neurons_0:", sum(different_task_list_neurons_0)/len(different_task_list_neurons_0))
print("different_task_list_neurons_3:", sum(different_task_list_neurons_3)/len(different_task_list_neurons_3))
print("different_task_list_neurons_6:", sum(different_task_list_neurons_6)/len(different_task_list_neurons_6))
if len(different_task_list_neurons_9)!=0:
    print("different_task_list_neurons_9:", sum(different_task_list_neurons_9)/len(different_task_list_neurons_9))
    print("different_task_list_neurons_12:", sum(different_task_list_neurons_12)/len(different_task_list_neurons_12))
if len(different_task_list_neurons_15)!=0:
    print("different_task_list_neurons_15:", sum(different_task_list_neurons_15)/len(different_task_list_neurons_15))
    print("different_task_list_neurons_18:", sum(different_task_list_neurons_18)/len(different_task_list_neurons_18))
    print("different_task_list_neurons_21:", sum(different_task_list_neurons_21)/len(different_task_list_neurons_21))
    print("different_task_list_neurons_24:", sum(different_task_list_neurons_24)/len(different_task_list_neurons_24))
