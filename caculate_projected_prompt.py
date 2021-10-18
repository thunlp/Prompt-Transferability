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


cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    return cos(task1_emb,task2_emb).sum()

def EuclideanDistances(task1_emb,task2_emb):
    return torch.norm(task1_emb-task2_emb, p='fro')
    '''
    sum_euclidence=0
    for i in range(len(task1_emb)):
        sum_euclidence += torch.norm(task1_emb[i]-task2_emb[i], p='fro')
    return sum_euclidence
    '''


def Euclidean(task1_emb, task2_emb):
    return torch.cdist(task1_emb,task2_emb,p=1)



root = "task_prompt_emb"

#task_map ={0:"IMDBPromptRoberta",1:"SST2PromptRoberta",2:"laptopPromptRoberta",3:"restaurantPromptRoberta",4:"movierationalesPromptRoberta",5:"tweetevalsentimentPromptRoberta",6:"MNLIPromptRoberta",7:"QNLIPromptRoberta",8:"WNLIPromptRoberta",9:"snliPromptRoberta",10:"recastnerPromptRoberta",11:"RTEPromptRoberta",12:"recastpunsPromptRoberta",13:"recastverbcornerPromptRoberta",14:"recastfactualityPromptRoberta",15:"recastmegaveridicalityPromptRoberta",16:"recastsentimentPromptRoberta",17:"recastverbnetPromptRoberta",18:"ethicscommonsensePromptRoberta",19:"ethicsdeontologyPromptRoberta",20:"ethicsjusticePromptRoberta",21:"ethicsvirtuePromptRoberta",22:"QQPPromptRoberta",23:"MRPCPromptRoberta"}

#task_map = {0:"IMDBPromptRoberta",1:"SST2PromptRoberta",2:"laptopPromptRoberta",3:"restaurantPromptRoberta",4:"movierationalesPromptRoberta",5:"tweetevalsentimentPromptRoberta",6:"MNLIPromptRoberta",7:"QNLIPromptRoberta",8:"snliPromptRoberta",9:"recastnerPromptRoberta",10:"ethicsdeontologyPromptRoberta",11:"ethicsjusticePromptRoberta",12:"QQPPromptRoberta",13:"MRPCPromptRoberta"}

task_map = ["IMDBPromptRoberta","laptopPromptRoberta","restaurantPromptRoberta","MNLIPromptRoberta","snliPromptRoberta"]

#sys.stdout = open("task_cos_distance.txt", 'w')
#sys.stdout = open("task_ecd_distance.txt", 'w')


print(end="\t")
#for id, name in task_map.items():
for name in task_map:
    #print(name, end='\t')
    name = name.replace("PromptRoberta","").replace("ethics","").replace("recast","")
    if len(name)>5:
        name = name[:5]
    print(name, end="\t")
print()


#for id_1, task_1 in task_map.items():
for task_1 in ["MNLIPromptRoberta","MNLI_base_nliPromptRoberta","IMDBPromptRoberta","IMDB_base_emotionPromptRoberta"]:
    #if id_1 not in show_in_list:
    #    continue
    cos_dict=dict()
    euc_dict=dict()
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

    name_1 = name_1.replace("PromptRoberta","").replace("ethics","").replace("recast","")
    if len(name_1)>5:
        name_1 = name_1[:5]
    print(name_1, end="\t")
    #for id_2, task_2 in task_map.items():
    for task_2 in task_map:
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

        #cos_dict[task_2]=float(CosineSimilarity(task_ten_1,task_ten_2))
        #sim=float(CosineSimilarity(task_ten_1,task_ten_2))

        #endcli
        #euc_dict[task_2]=float(EuclideanDistances(task_ten_1,task_ten_2))
        sim=float(EuclideanDistances(task_ten_1,task_ten_2))


        #print(sim, end='\t')
        #print("{:.2f}".format(float(sim)), end='\t')
        print("{:.0f}".format(float(sim)), end='\t')

    print()



