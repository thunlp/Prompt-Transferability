import argparse
import logging
import random
import numpy as np
import os
import json
import math
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
import copy


#from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
#from openTSNE import initialization
#from openTSNE.callbacks import ErrorLogger
#from examples import utils
#from openTSNE_.examples import utils_
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import matplotlib.lines as mlines

#from tsnecuda import TSNE
import glob

#####





def PCA_svd(X=None, k=None, center=True):
    #############original##############
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h

    H = H.cpu()
    X = X.cpu()

    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    s_2 = s*s
    print("Compression rate: {}% ".format( int(torch.sqrt(s_2[:k].sum()/s_2.sum())*100) ) )
    #exit()
    return components
    ###################################


    #############Analysis##############
    #task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}

    '''
    #sentiment
    indices = torch.tensor([0,9,10,11])
    X = torch.index_select(X, 0, indices)
    #X = sentiment

    #NLI : [1,3,7]
    indices = torch.tensor([1,3,7])
    X = torch.index_select(X, 0, indices)
    #X = NLI


    #mix
    indices = torch.tensor([0,9,10,11,1,3,7])
    X = torch.index_select(X, 0, indices)

    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    #H = H.cuda()
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    s_2 = s*s
    print("Compression rate: {}% ".format( int(torch.sqrt(s_2[:k].sum()/s_2.sum())*100) ) )
    ####################################
    '''


    return components



#def PCA(input=None, out_features=None):
#    return torch.pca_lowrank(x[i],q=3)[1]






task_define ={"IMDB":"emotion","laptop":"emotion","MNLI":"nli","movierationales":"emotion","MRPC":"sentence_sim","QNLI":"nli","QQP":"sentence_sim","restaurant":"emotion","snli":"nli","SST2":"emotion","tweetevalsentiment":"emotion","ethicsdeontology":"ethics","ethicsjustice":"ethics","recastner":"nli"}



all_prompt_emb = list()
for task ,ty in task_define.items():
    path="task_prompt_emb/"+str(task)+"PromptRoberta/task_prompt"
    ten = torch.load(path,map_location=lambda storage, loc: storage).view(76800)
    all_prompt_emb.append(ten)

all_prompt_emb = torch.stack(all_prompt_emb)
print(all_prompt_emb.shape)





#PCA compress
####################
dim=2
compressed_prompt_emb = PCA_svd(X=all_prompt_emb,k=dim)
print(compressed_prompt_emb.shape)

#task 14
#all: 88%

####################



##################
#######Draw
##################
compressed_prompt_emb = compressed_prompt_emb.to("cpu").detach().numpy()
#all_label = all_label.to("cpu").numpy()

#color table: https://www.computerhope.com/htmcolor.htm#color-codes

#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}
###
#task_map={0:"sst2",1:"re",2:"laptop",3:"restaurant",4:"IMDB"}

#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"QNLI",6:"#CB6D51",7:"#556B2F",8:"STSB",9:"#4863A0",10:"#151B8D"}

#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"#64CD64",6:"#CB6D51",7:"#556B2F",8:"#FFC0CB",9:"#4863A0",10:"#151B8D",11:"#00FFFF"}

color_map={"emotion":"#A9CCE3","nli":"#B6D7A8","ethics":"#F5CBA7","sentence_sim":"#D5DBDB"}

#re generate id
#plot on 3D: https://www.delftstack.com/zh-tw/howto/matplotlib/scatter-plot-legend-in-matplotlib/#%25E5%259C%25A8-matplotlib-3d-%25E6%2595%25A3%25E9%25BB%259E%25E5%259C%2596%25E4%25B8%258A%25E6%2596%25B0%25E5%25A2%259E%25E5%259C%2596%25E4%25BE%258B


if dim ==3 :
    axes = plt.subplot(111, projection='3d')
    #axes = plt.subplot(222, projection='3d')
elif dim == 2:
    #axes = plt.subplot(111, projection='2d')
    fig, axes = plt.subplots()

else:
    pass


task_list = list()

#for task_id, task_name in task_map.items():
for task_id, task_name in enumerate(task_define):
    task_type = task_define[task_name]
    task_color = color_map[task_type]

    if task_name =="IMDB":
        task_name = "imdb"
    elif task_name =="SST2":
        task_name = "sst2"
    elif task_name =="laptop":
        task_name = "laptop"
    elif task_name =="restaurant":
        pass
    elif task_name =="movierationales":
        task_name = "movie"
    elif task_name =="tweetevalsentiment":
        task_name = "tweet"
    elif task_name =="MNLI":
        task_name = "mnli"
    elif task_name =="QNLI":
        task_name = "qnli"
    elif task_name =="snli":
        task_name = "snli"
    elif task_name =="recastner":
        task_name = "ner"
    elif task_name =="ethicsdeontology":
        task_name = "deontology"
    elif task_name =="ethicsjustice":
        task_name = "justics"
    elif task_name =="QQP":
        task_name = "qqp"
    elif task_name =="MRPC":
        task_name = "mrpc"

    #task_list.append(task_name)


    print(task_name)


    #if task_id in blocked_list:
    #    continue

    ###
    #axes.plot(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1],"o", color=task_color)
    #axes.text(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], task_name)

    plt.scatter(x=compressed_prompt_emb[task_id][0], y=compressed_prompt_emb[task_id][1], s=280, marker="o", color=task_color)
    plt.text(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], task_name, fontsize=12)
    ###


#plt.legend(labels=task_list, loc='best')

x_list = list()
y_list = list()
for x_ , y_ in color_map.items():
    x_list.append(x_)
    y_list.append(y_)


#color_map={"emotion":"#A9CCE3","nli":"#B6D7A8","ethics":"#F5CBA7","sentence_sim":"#D5DBDB"}
#plt.legend(labels=[x_list], loc='best')


emotion = mlines.Line2D([], [], color="#A9CCE3", marker='o',markersize=24, label='emotion', linewidth=0, linestyle=None)
nli = mlines.Line2D([], [], color="#B6D7A8", marker='o',markersize=24, label='nli', linewidth=0, linestyle=None)
ethics = mlines.Line2D([], [], color="#F5CBA7", marker='o',markersize=24, label='ethics', linewidth=0, linestyle=None)
sentence_pair = mlines.Line2D([], [], color="#D5DBDB", marker='o',markersize=24, label='sentence_pair', linewidth=0, linestyle=None)
blank = mlines.Line2D([], [], color="#FFFFFF", marker='o',markersize=1, label='', linewidth=0, linestyle=None)

plt.legend(handles=[blank, emotion, blank, nli, blank, ethics, blank, sentence_pair, blank], loc="best")
#plt.legend(handles=[red_dot], loc="best")


#plt.legend()
#plt.legend(loc="upper left")
#plt.title("Task Prompt Dist")
#plt.savefig('output.pdf')
plt.savefig('task_prompt_emb_in_2D.pdf', bbox_inches='tight')
#plt.savefig('output.jpg')
#plt.savefig('exp_results/SENTIMENT.jpg')
#plt.savefig('exp_results/NLI.jpg')
#plt.savefig('exp_results/domain.jpg')
#plt.savefig('exp_results/PCA_DOMAIN_BASE_SENTIMENT.jpg')


