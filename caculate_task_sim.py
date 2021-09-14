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


'''
prefiex=8
sst_extra_ten=dict()
sst_task_map=dict()
for i in range(1,15):
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta_"+str(i)+"/"
    sst2_file = os.listdir(path)[0]
    sst2_ten = torch.load(path+sst2_file)
    #print(sst2_ten.shape)
    sst_extra_ten[prefiex+i] = sst2_ten
    sst_task_map[prefiex+i] = "sst2_"+str(i)
'''

'''
#restaurant
restaurant_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/restaurantPromptRoberta/"
restaurant_file = os.listdir(path)[0]
restaurant_ten=torch.load(path+restaurant_file)
#sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]*sst2_ten.shape[1]))
#sst2_ten = sst2_ten[0]
print(restaurant_ten.shape)


#laptop
laptop_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/laptopPromptRoberta/"
laptop_file = os.listdir(path)[0]
laptop_ten=torch.load(path+laptop_file)
#sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]*sst2_ten.shape[1]))
#sst2_ten = sst2_ten[0]
print(laptop_ten.shape)


#SST2
sst2_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/"
sst2_file = os.listdir(path)[0]
sst2_ten=torch.load(path+sst2_file)
#sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]*sst2_ten.shape[1]))
#sst2_ten = sst2_ten[0]
print(sst2_ten.shape)


#SST2
sst2_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/"
sst2_file = os.listdir(path)[0]
sst2_ten=torch.load(path+sst2_file)
#sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]*sst2_ten.shape[1]))
#sst2_ten = sst2_ten[0]
print(sst2_ten.shape)


#RTE
rte_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/RTEPromptRoberta/"
rte_file = os.listdir(path)[0]
rte_ten=torch.load(path+rte_file)
#rte_ten = rte_ten.reshape(int(rte_ten.shape[0]*rte_ten.shape[1]))
print(rte_ten.shape)
#exit()


#RE
re_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/REPrompt/"
re_file = os.listdir(path)[0]
re_ten=torch.load(path+re_file)
#re_ten = re_ten.reshape(int(re_ten.shape[0]*re_ten.shape[1]))
print(re_ten.shape)
#exit()

###

#MNLI
MNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MNLIPromptRoberta/"
MNLI_file = os.listdir(path)[0]
MNLI_ten=torch.load(path+MNLI_file)
#MNLI_ten = MNLI_ten.reshape(int(MNLI_ten.shape[0]*MNLI_ten.shape[1]))
print(MNLI_ten.shape)


#MRPC
MRPC_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MRPCPromptRoberta/"
MRPC_file = os.listdir(path)[0]
MRPC_ten=torch.load(path+MRPC_file)
#MRPC_ten = MRPC_ten.reshape(int(MRPC_ten.shape[0]*MRPC_ten.shape[1]))
print(MRPC_ten.shape)



#QNLI
QNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QNLIPromptRoberta/"
QNLI_file = os.listdir(path)[0]
QNLI_ten=torch.load(path+QNLI_file)
#QNLI_ten = QNLI_ten.reshape(int(QNLI_ten.shape[0]*QNLI_ten.shape[1]))
print(QNLI_ten.shape)


#QQP
QQP_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QQPPromptRoberta/"
QQP_file = os.listdir(path)[0]
QQP_ten=torch.load(path+QQP_file)
#QQP_ten = QQP_ten.reshape(int(QQP_ten.shape[0]*QQP_ten.shape[1]))
print(QQP_ten.shape)


#WNLI
WNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/WNLIPromptRoberta/"
WNLI_file = os.listdir(path)[0]
WNLI_ten=torch.load(path+WNLI_file)
#WNLI_ten = WNLI_ten.reshape(int(WNLI_ten.shape[0]*WNLI_ten.shape[1]))
print(WNLI_ten.shape)


#STSB
STSB_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/STSBPromptRoberta/"
STSB_file = os.listdir(path)[0]
STSB_ten=torch.load(path+STSB_file)
#STSB_ten = STSB_ten.reshape(int(STSB_ten.shape[0]*STSB_ten.shape[1]))
print(STSB_ten.shape)

###########################
###########################
###########################

task_ten= {0:sst2_ten,1:rte_ten,2:re_ten,3:MNLI_ten,4:MRPC_ten,5:QNLI_ten,6:QQP_ten,7:WNLI_ten,8:STSB_ten,9:laptop_ten,10:restaurant_ten}
#task_ten.update(sst_extra_ten)
'''

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    return cos(task1_emb,task2_emb).sum()

def EuclideanDistances(task1_emb,task2_emb):
    #return torch.norm(task1_emb-task2_emb, p='fro')
    sum_euclidence=0
    for i in range(len(task1_emb)):
        sum_euclidence += torch.norm(task1_emb[i]-task2_emb[i], p='fro')
    return sum_euclidence


def Euclidean(task1_emb, task2_emb):
    return torch.cdist(task1_emb,task2_emb,p=1)



root = "task_prompt_emb"

task_map = {0:"IMDB",1:"SST2",2:"laptop",3:"rest",4:"movie",5:"tweet",6:"MNLI",7:"QNLI",8:"WNLI",9:"snli",10:"RTE",11:"QQP",12:"MRPC"}


#sys.stdout = open("task_cos_distance.txt", 'w')
#sys.stdout = open("task_ecd_distance.txt", 'w')


print(end="\t")
for id, name in task_map.items():
    print(name, end='\t')
print()


for id_1, task_1 in task_map.items():
    #if id_1 not in show_in_list:
    #    continue
    cos_dict=dict()
    euc_dict=dict()
    print(task_1, end="\t")
    if task_1 == "rest":
        name_1 = "restaurant"
    elif task_1 == "movie":
        name_1 = "movierationales"
    elif task_1 == "tweet":
        name_1 = "tweetevalsentiment"
    else:
        name_1 = task_1
    task_ten_1 = torch.load(root+"/"+name_1+"PromptRoberta/task_prompt", map_location=lambda storage, loc: storage)
    task_ten_1 = task_ten_1.reshape(task_ten_1.shape[0]*task_ten_1.shape[1])
    for id_2, task_2 in task_map.items():
        #if id_2 not in show_in_list:
        #    continue
        #if id_1 == id_2:
        #    continue
        #else:
        #similiarty:
        #cos:
        if task_2 == "rest":
            name_2 = "restaurant"
        elif task_2 == "movie":
            name_2 = "movierationales"
        elif task_2 == "tweet":
            name_2 = "tweetevalsentiment"
        else:
            name_2 = task_2
        task_ten_2 = torch.load(root+"/"+name_2+"PromptRoberta/task_prompt", map_location=lambda storage, loc: storage)
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



