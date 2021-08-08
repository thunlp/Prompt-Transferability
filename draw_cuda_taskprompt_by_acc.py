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
from openTSNE_.examples import utils_
#import utils
import numpy as np
import matplotlib.pyplot as plt

from tsnecuda import TSNE
#from os import listdir
#from os.path import isfile, join
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


def plot(x, y, **kwargs):
    utils_.plot(
        x,
        y,
        colors=utils_.MOUSE_10X_COLORS,
        alpha=kwargs.pop("alpha", 0.1),
        draw_legend=True,
        draw_centers=True,
        #dot_size=20,
        label_map=kwargs["label_map"],
        **kwargs,
    )





#IMDB
IMDB_ten = list()
path="result/IMDBPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        IMDB_ten.append(float(json.loads(file)["acc"])*100)
IMDB_ten = torch.tensor(IMDB_ten)


#SST2
sst2_ten = list()
path="result/SST2PromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        sst2_ten.append(float(json.loads(file)["acc"])*100)
sst2_ten = torch.tensor(sst2_ten)


#laptop
laptop_ten = list()
path="result/laptopPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        laptop_ten.append(float(json.loads(file)["acc"])*100)
laptop_ten = torch.tensor(laptop_ten)


#restaurant
restaurant_ten = list()
path="result/restaurantPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        restaurant_ten.append(float(json.loads(file)["acc"])*100)
restaurant_ten = torch.tensor(restaurant_ten)

#############
#############


#RTE
rte_ten = list()
path="result/RTEPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        rte_ten.append(float(json.loads(file)["acc"])*100)
rte_ten = torch.tensor(rte_ten)


#MNLI
MNLI_ten = list()
path="result/MNLIPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        MNLI_ten.append(float(json.loads(file)["acc"])*100)
MNLI_ten = torch.tensor(MNLI_ten)
print(MNLI_ten)
print(MNLI_ten.shape)



#WNLI
WNLI_ten = list()
path="result/WNLIPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        WNLI_ten.append(float(json.loads(file)["acc"])*100)
WNLI_ten = torch.tensor(WNLI_ten)
print(WNLI_ten)
print(WNLI_ten.shape)


################
#Paraphrase
################

#MRPC
MRPC_ten = list()
path="result/MRPCPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        MRPC_ten.append(float(json.loads(file)["acc"])*100)
MRPC_ten = torch.tensor(MRPC_ten)
print(MRPC_ten)
print(MRPC_ten.shape)


#QQP
QQP_ten = list()
path="result/QQPPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        QQP_ten.append(float(json.loads(file)["acc"])*100)
QQP_ten = torch.tensor(QQP_ten)
print(QQP_ten)
print(QQP_ten.shape)


################
#RE
################
#RE



################
#Other
################


#QNLI
QNLI_ten = list()
path="result/QNLIPromptRoberta"
for prompt_model in ["IMDBPromptRoberta","laptopPromptRoberta","MNLIPromptRoberta","MRPCPromptRoberta","QNLIPromptRoberta","QQPPromptRoberta","restaurantPromptRoberta","RTEPromptRoberta","SST2PromptRoberta","WNLIPromptRoberta"]:
    with open(path+"/result_"+prompt_model+".json") as file:
        file = json.load(file)
        QNLI_ten.append(float(json.loads(file)["acc"])*100)
QNLI_ten = torch.tensor(QNLI_ten)
print(QNLI_ten)
print(QNLI_ten.shape)



#STSB




###########################
###########################
###########################

#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}
#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}
task_map={0:"sst2",1:"rte",2:"MNLI",3:"MRPC",4:"QNLI",5:"QQP",6:"WNLI",7:"laptop",8:"restaurant",9:"IMDB"}

#92%
#all_prompt_emb = torch.stack([sst2_ten,rte_ten,re_ten,MNLI_ten,MRPC_ten,QNLI_ten,QQP_ten,WNLI_ten,STSB_ten,laptop_ten,restaurant_ten,IMDB_ten])
###
all_prompt_emb = torch.stack([sst2_ten,rte_ten,MNLI_ten,MRPC_ten,QNLI_ten,QQP_ten,WNLI_ten,laptop_ten,restaurant_ten,IMDB_ten])

print("===================")
print("===================")

##3D or 2D
#dim=3
dim=2
#compressed_prompt_emb = train_AE(input=all_prompt_emb,out_features=dim)





#PCA compress
####################
compressed_prompt_emb = PCA_svd(X=all_prompt_emb,k=dim)
print(compressed_prompt_emb.shape)

#all: 92%
#sentiment: 100%

####################



##################
#######Draw
##################
compressed_prompt_emb = compressed_prompt_emb.to("cpu").detach().numpy()
#all_label = all_label.to("cpu").numpy()

#color table: https://www.computerhope.com/htmcolor.htm#color-codes

#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}
task_map={0:"sst2",1:"rte",2:"MNLI",3:"MRPC",4:"QNLI",5:"QQP",6:"WNLI",7:"laptop",8:"restaurant",9:"IMDB"}
###
#task_map={0:"sst2",1:"re",2:"laptop",3:"restaurant",4:"IMDB"}

#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"QNLI",6:"#CB6D51",7:"#556B2F",8:"STSB",9:"#4863A0",10:"#151B8D"}
#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"#64CD64",6:"#CB6D51",7:"#556B2F",8:"#FFC0CB",9:"#4863A0",10:"#151B8D",11:"#00FFFF"}
color_map={0:"#728FCE",1:"#347235",2:"#6B8E23",3:"#C04000",4:"#64CD64",5:"#CB6D51",6:"#556B2F",7:"#4863A0",8:"#151B8D",9:"#00FFFF"}


blocked_list = []
#blocked_list = [1,3,4,5,6,7,8]
#blocked_list = [1,3,2,5,8]

#sentiment, NLI, RE, Paraphrase
#blocked_list = [5,8]


#========================
#Domain

#Wiki, reataurant, computer, movie, Fiction
#blocked_list = [3,6,8]


#========================
#Task

#NLI, RE, sentiment
#blocked_list = [4,5,6,8]

#NLI, RE
#blocked_list = [0,8,9,10,11]



#NLI, RE, Paraphrase
#blocked_list = [0,5,9,10]

#NLI, RE, Paraphrase
#blocked_list = [0,5,8,9,10]

#sentiment, RE
#blocked_list = [5,8,3,4,5,6,7,8,2]

#sentiment
#blocked_list = [5,8,3,4,5,6,7,8,2,1]

#sentiment, NLI
#blocked_list = [2,4,5,6,8]

#sentiment, Paraphrase
#blocked_list = [1,2,3,5,7,8]

#NLI, Paraphrase
#blocked_list = [0,1,2,5,8,9,10]



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

for task_id, task_name in task_map.items():
    print(task_id)
    if task_id in blocked_list:
        continue
    #try:
    print(compressed_prompt_emb[task_id])
    #except:
    #    continue

    if dim == 2:
        ###
        #label_map
        '''
        plt.scatter(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], color=color_map[task_id], label=task_map[task_id], s=100)
        '''
        #text on dot
        axes.plot(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1],"o", color=color_map[task_id])
        #axes.annotate(task_map[task_id],(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], compressed_prompt_emb[task_id][2]))
        axes.text(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], task_map[task_id])
        ###
    elif dim == 3:
        ###
        #label_map
        '''
        axes.plot(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], compressed_prompt_emb[task_id][2], "o", color=color_map[task_id], label=task_map[task_id])
        '''

        #text on dot
        axes.plot(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], compressed_prompt_emb[task_id][2], "o", color=color_map[task_id])
        axes.text(compressed_prompt_emb[task_id][0], compressed_prompt_emb[task_id][1], compressed_prompt_emb[task_id][2], task_map[task_id])
        ###
    else:
        print("Wonrg!!!")
        exit()

'''
if dim == 2:
    plt.legend()
elif dim == 3:
    pass
'''


#plt.legend()
#plt.legend(loc="upper left")
#plt.title("Task Prompt Dist")
#plt.savefig('output.pdf')
plt.savefig('output.jpg')
#plt.savefig('exp_results/SENTIMENT.jpg')
#plt.savefig('exp_results/NLI.jpg')
#plt.savefig('exp_results/domain.jpg')
#plt.savefig('exp_results/PCA_DOMAIN_BASE_SENTIMENT.jpg')


