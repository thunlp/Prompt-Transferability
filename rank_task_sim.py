import argparse
import logging
import random
import numpy as np
import os
import json
import math
import numpy

import torch


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



#SST2
sst2_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/"
sst2_file = os.listdir(path)[0]
#print(sst2_file)
#exit()
#for file in sst2_file:
    #print(torch.load(path+file).shape)
    #sst2_ten.append(torch.load(path+file))
#sst2_ten = torch.cat(sst2_ten)
sst2_ten=torch.load(path+sst2_file)
#print(sst2_ten.shape)
#exit()
#sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]),int(sst2_ten.shape[1]*sst2_ten.shape[2]))
sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]*sst2_ten.shape[1]))
#sst2_ten = sst2_ten[0]
print(sst2_ten.shape)


#RTE
rte_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/RTEPromptRoberta/"
rte_file = os.listdir(path)[0]
#print(rte_file)
#for file in rte_file:
    #print(torch.load(path+file).shape)
#    rte_ten.append(torch.load(path+file))
#rte_ten = torch.cat(rte_ten)
#print(rte_ten.shape)
#rte_ten = rte_ten.reshape(int(rte_ten.shape[0]),int(rte_ten.shape[1]*rte_ten.shape[2]))
rte_ten=torch.load(path+rte_file)
rte_ten = rte_ten.reshape(int(rte_ten.shape[0]*rte_ten.shape[1]))
#rte_ten = rte_ten[0]
print(rte_ten.shape)
#exit()


#RE
re_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/REPrompt/"
re_file = os.listdir(path)[0]
#print(re_file)
#for file in re_file:
    #print(torch.load(path+file).shape)
#    re_ten.append(torch.load(path+file))
#re_ten = torch.cat(re_ten)
#print(re_ten.shape)
#re_ten = re_ten.reshape(int(re_ten.shape[0]),int(re_ten.shape[1]*re_ten.shape[2]))
#re_ten = re_ten[0]
re_ten=torch.load(path+re_file)
re_ten = re_ten.reshape(int(re_ten.shape[0]*re_ten.shape[1]))
print(re_ten.shape)
#exit()

###

#MNLI
MNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MNLIPromptRoberta/"
MNLI_file = os.listdir(path)[0]
MNLI_ten=torch.load(path+MNLI_file)
MNLI_ten = MNLI_ten.reshape(int(MNLI_ten.shape[0]*MNLI_ten.shape[1]))
print(MNLI_ten.shape)


#MRPC
MRPC_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MRPCPromptRoberta/"
MRPC_file = os.listdir(path)[0]
MRPC_ten=torch.load(path+MRPC_file)
MRPC_ten = MRPC_ten.reshape(int(MRPC_ten.shape[0]*MRPC_ten.shape[1]))
print(MRPC_ten.shape)



#QNLI
QNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QNLIPromptRoberta/"
QNLI_file = os.listdir(path)[0]
QNLI_ten=torch.load(path+QNLI_file)
QNLI_ten = QNLI_ten.reshape(int(QNLI_ten.shape[0]*QNLI_ten.shape[1]))
print(QNLI_ten.shape)


#QQP
QQP_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QQPPromptRoberta/"
QQP_file = os.listdir(path)[0]
QQP_ten=torch.load(path+QQP_file)
QQP_ten = QQP_ten.reshape(int(QQP_ten.shape[0]*QQP_ten.shape[1]))
print(QQP_ten.shape)


#WNLI
WNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/WNLIPromptRoberta/"
WNLI_file = os.listdir(path)[0]
WNLI_ten=torch.load(path+WNLI_file)
WNLI_ten = WNLI_ten.reshape(int(WNLI_ten.shape[0]*WNLI_ten.shape[1]))
print(WNLI_ten.shape)


#STSB
STSB_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/STSBPromptRoberta/"
STSB_file = os.listdir(path)[0]
STSB_ten=torch.load(path+STSB_file)
STSB_ten = STSB_ten.reshape(int(STSB_ten.shape[0]*STSB_ten.shape[1]))
print(STSB_ten.shape)

###########################
###########################
###########################

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    return cos(task1_emb,task2_emb)

def EuclideanDistances(task1_emb,task2_emb):
    return torch.norm(task1_emb-task2_emb, p='fro')


def Euclidean(task1_emb, task2_emb):
    return torch.cdist(task1_emb,task2_emb,p=1)

task_ten={0:sst2_ten,1:rte_ten,2:re_ten,3:MNLI_ten,4:MRPC_ten,5:QNLI_ten,6:QQP_ten,7:WNLI_ten,8:STSB_ten}
task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB"}


for id_1, task_1 in task_map.items():
    cos_dict=dict()
    euc_dict=dict()
    for id_2, task_2 in task_map.items():
        if id_1 == id_2:
            continue
        else:
            #similiarty:
            #cos:
            cos_dict[task_map[id_2]]=float(CosineSimilarity(task_ten[id_1],task_ten[id_2]))

            #endcli
            euc_dict[task_map[id_2]]=float(EuclideanDistances(task_ten[id_1],task_ten[id_2]))

    #ranking
    print("=======================")
    print("==",task_1,"==")
    print("-------")
    print("CosineSimilarity:")
    print("-------")
    for task_2 in sorted(cos_dict, key=cos_dict.get, reverse=True):
        print(task_2, cos_dict[task_2])

    print("-------")
    print("EuclideanDistances:")
    print("-------")
    for task_2 in sorted(euc_dict, key=euc_dict.get, reverse=False):
        print(task_2, euc_dict[task_2])

    print("=======================")



