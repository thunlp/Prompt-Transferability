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
    #H = H.cuda()
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
path="task_prompt_emb/IMDBPromptRoberta/task_prompt"
IMDB_ten = torch.load(path).view(76800)
print(IMDB_ten.shape)
#IMDB_ten = torch.stack([IMDB_ten for i in range(200)])
#print(IMDB_ten.shape)



#SST2
sst2_ten = list()
path="task_prompt_emb/SST2PromptRoberta/task_prompt"
sst2_ten = torch.load(path).view(76800)
print(sst2_ten.shape)
#sst2_ten = torch.stack([sst2_ten for i in range(200)])
#print(sst2_ten.shape)


#laptop
laptop_ten = list()
laptop_ten = list()
path="task_prompt_emb/laptopPromptRoberta/task_prompt"
laptop_ten = torch.load(path).view(76800)
print(laptop_ten.shape)
#laptop_ten = torch.stack([laptop_ten for i in range(200)])
#print(laptop_ten.shape)


#restaurant
restaurant_ten = list()
restaurant_ten = list()
path="task_prompt_emb/restaurantPromptRoberta/task_prompt"
restaurant_ten = torch.load(path).view(76800)
print(restaurant_ten.shape)
#restaurant_ten = torch.stack([restaurant_ten for i in range(200)])
#print(restaurant_ten.shape)

#############
#############


#RTE
rte_ten = list()
path="task_prompt_emb/RTEPromptRoberta/task_prompt"
rte_ten = torch.load(path).view(76800)
print(rte_ten.shape)
#rte_ten = torch.stack([rte_ten for i in range(200)])
#print(rte_ten.shape)


#MNLI
MNLI_ten = list()
MNLI_ten = list()
path="task_prompt_emb/MNLIPromptRoberta/task_prompt"
MNLI_ten = torch.load(path).view(76800)
print(MNLI_ten.shape)
#MNLI_ten = torch.stack([MNLI_ten for i in range(200)])
#print(MNLI_ten.shape)



#WNLI
WNLI_ten = list()
WNLI_ten = list()
path="task_prompt_emb/WNLIPromptRoberta/task_prompt"
WNLI_ten = torch.load(path).view(76800)
print(WNLI_ten.shape)
#WNLI_ten = torch.stack([WNLI_ten for i in range(200)])
#print(WNLI_ten.shape)


################
#Paraphrase
################

#MRPC
MRPC_ten = list()
MRPC_ten = list()
path="task_prompt_emb/MRPCPromptRoberta/task_prompt"
MRPC_ten = torch.load(path).view(76800)
print(MRPC_ten.shape)
#MRPC_ten = torch.stack([MRPC_ten for i in range(200)])
#print(MRPC_ten.shape)


#QQP
QQP_ten = list()
QQP_ten = list()
path="task_prompt_emb/QQPPromptRoberta/task_prompt"
QQP_ten = torch.load(path).view(76800)
print(QQP_ten.shape)
#QQP_ten = torch.stack([QQP_ten for i in range(200)])
#print(QQP_ten.shape)


################
#RE
################
#RE
re_ten = list()
path="task_prompt_emb/REPrompt/task_prompt"
re_ten = torch.load(path).view(76800)
print(re_ten.shape)
#re_ten = torch.stack([re_ten for i in range(200)])
#print(re_ten.shape)


################
#Other
################


#QNLI
QNLI_ten = list()
QNLI_ten = list()
path="task_prompt_emb/QNLIPromptRoberta/task_prompt"
QNLI_ten = torch.load(path).view(76800)
print(QNLI_ten.shape)
#QNLI_ten = torch.stack([QNLI_ten for i in range(200)])
#print(QNLI_ten.shape)



#STSB
STSB_ten = list()
STSB_ten = list()
path="task_prompt_emb/STSBPromptRoberta/task_prompt"
STSB_ten = torch.load(path).view(76800)
print(STSB_ten.shape)
#STSB_ten = torch.stack([STSB_ten for i in range(200)])
#print(STSB_ten.shape)




########################################################
######################BERT##############################
########################################################


#IMDB
IMDB_ten_B = list()
path="task_prompt_emb/IMDBPromptBert/task_prompt"
IMDB_ten_B = torch.load(path).view(76800)
print(IMDB_ten_B.shape)
#IMDB_ten_B = torch.stack([IMDB_ten_B for i in range(200)])
#print(IMDB_ten_B.shape)


'''
#SST2
sst2_ten_B = list()
path="task_prompt_emb/SST2PromptBert/task_prompt"
sst2_ten_B = torch.load(path).view(76800)
print(sst2_ten_B.shape)
#sst2_ten_B = torch.stack([sst2_ten_B for i in range(200)])
#print(sst2_ten_B.shape)
'''


#laptop
laptop_ten_B = list()
path="task_prompt_emb/laptopPromptBert/task_prompt"
laptop_ten_B = torch.load(path).view(76800)
print(laptop_ten_B.shape)
#laptop_ten_B = torch.stack([laptop_ten_B for i in range(200)])
#print(laptop_ten_B.shape)


#restaurant
restaurant_ten_B = list()
path="task_prompt_emb/restaurantPromptBert/task_prompt"
restaurant_ten_B = torch.load(path).view(76800)
print(restaurant_ten_B.shape)
#restaurant_ten_B = torch.stack([restaurant_ten_B for i in range(200)])
#print(restaurant_ten_B.shape)

#############
#############


'''
#RTE
rte_ten_B = list()
path="task_prompt_emb/RTEPromptBert/task_prompt"
rte_ten_B = torch.load(path).view(76800)
print(rte_ten_B.shape)
#rte_ten_B = torch.stack([rte_ten_B for i in range(200)])
#print(rte_ten_B.shape)


#MNLI
MNLI_ten_B = list()
path="task_prompt_emb/MNLIPromptBert/task_prompt"
MNLI_ten_B = torch.load(path).view(76800)
print(MNLI_ten_B.shape)
#MNLI_ten_B = torch.stack([MNLI_ten_B for i in range(200)])
#print(MNLI_ten_B.shape)



#WNLI
WNLI_ten_B = list()
path="task_prompt_emb/WNLIPromptBert/task_prompt"
WNLI_ten_B = torch.load(path).view(76800)
print(WNLI_ten_B.shape)
#WNLI_ten_B = torch.stack([WNLI_ten_B for i in range(200)])
#print(WNLI_ten_B.shape)


################
#Paraphrase
################

#MRPC
MRPC_ten_B = list()
MRPC_ten_B = list()
path="task_prompt_emb/MRPCPromptBert/task_prompt"
MRPC_ten_B = torch.load(path).view(76800)
print(MRPC_ten_B.shape)
#MRPC_ten_B = torch.stack([MRPC_ten_B for i in range(200)])
#print(MRPC_ten_B.shape)


#QQP
QQP_ten_B = list()
path="task_prompt_emb/QQPPromptBert/task_prompt"
QQP_ten_B = torch.load(path).view(76800)
print(QQP_ten_B.shape)
#QQP_ten_B = torch.stack([QQP_ten_B for i in range(200)])
#print(QQP_ten_B.shape)


################
#RE
################
#RE
re_ten_B = list()
path="task_prompt_emb/REPrompt/task_prompt"
re_ten_B = torch.load(path).view(76800)
print(re_ten_B.shape)
#re_ten_B = torch.stack([re_ten_B for i in range(200)])
#print(re_ten_B.shape)


################
#Other
################


#QNLI
QNLI_ten_B = list()
path="task_prompt_emb/QNLIPromptBert/task_prompt"
QNLI_ten_B = torch.load(path).view(76800)
print(QNLI_ten_B.shape)
#QNLI_ten_B = torch.stack([QNLI_ten_B for i in range(200)])
#print(QNLI_ten_B.shape)



#STSB
STSB_ten_B = list()
path="task_prompt_emb/STSBPromptBert/task_prompt"
STSB_ten_B = torch.load(path).view(76800)
print(STSB_ten_B.shape)
#STSB_ten_B = torch.stack([STSB_ten_B for i in range(200)])
#print(STSB_ten_B.shape)
'''





######################################################
######################################################
######################################################

#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}

task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB",12:"sst2_B",13:"rte_B",14:"re_B",15:"MNLI_B",16:"MRPC_B",17:"QNLI_B",18:"QQP_B",19:"WNLI_B",20:"STSB_B",21:"laptop_B",22:"restaurant_B",23:"IMDB_B"}

#task_map={0:"sst2",1:"rte",2:"re"}

#0
sst2_label_ten = torch.zeros(int(sst2_ten.shape[0]),dtype=torch.int64)
#1
rte_label_ten = torch.ones(int(rte_ten.shape[0]),dtype=torch.int64)
#2
re_label_ten = torch.ones(int(re_ten.shape[0]),dtype=torch.int64)
re_label_ten[re_label_ten==1]=2
#3
MNLI_label_ten = torch.ones(int(MNLI_ten.shape[0]),dtype=torch.int64)
MNLI_label_ten[MNLI_label_ten==1]=3
#4
MRPC_label_ten = torch.ones(int(MRPC_ten.shape[0]),dtype=torch.int64)
MRPC_label_ten[MRPC_label_ten==1]=4
#5
QNLI_label_ten = torch.ones(int(QNLI_ten.shape[0]),dtype=torch.int64)
QNLI_label_ten[QNLI_label_ten==1]=5
#6
QQP_label_ten = torch.ones(int(QQP_ten.shape[0]),dtype=torch.int64)
QQP_label_ten[QQP_label_ten==1]=6
#7
WNLI_label_ten = torch.ones(int(WNLI_ten.shape[0]),dtype=torch.int64)
WNLI_label_ten[WNLI_label_ten==1]=7
#8
STSB_label_ten = torch.ones(int(STSB_ten.shape[0]),dtype=torch.int64)
STSB_label_ten[STSB_label_ten==1]=8
#9
laptop_label_ten = torch.ones(int(laptop_ten.shape[0]),dtype=torch.int64)
laptop_label_ten[laptop_label_ten==1]=9
#10
restaurant_label_ten = torch.ones(int(restaurant_ten.shape[0]),dtype=torch.int64)
restaurant_label_ten[restaurant_label_ten==1]=10
#11
IMDB_label_ten = torch.ones(int(IMDB_ten.shape[0]),dtype=torch.int64)
IMDB_label_ten[IMDB_label_ten==1]=11

################################################
################################################

'''
#12
sst2_label_ten_B = torch.ones(int(sst2_ten_B.shape[0]),dtype=torch.int64)
sst2_label_ten_B[sst2_label_ten_B==1]=12
#13
rte_label_ten_B = torch.ones(int(rte_ten_B.shape[0]),dtype=torch.int64)
rte_label_ten_B[rte_label_ten_B==1]=13
#14
re_label_ten_B = torch.ones(int(re_ten_B.shape[0]),dtype=torch.int64)
re_label_ten_B[re_label_ten_B==1]=14
#15
MNLI_label_ten_B = torch.ones(int(MNLI_ten_B.shape[0]),dtype=torch.int64)
MNLI_label_ten_B[MNLI_label_ten_B==1]=15
#16
MRPC_label_ten_B = torch.ones(int(MRPC_ten_B.shape[0]),dtype=torch.int64)
MRPC_label_ten_B[MRPC_label_ten_B==1]=16
#17
QNLI_label_ten_B = torch.ones(int(QNLI_ten_B.shape[0]),dtype=torch.int64)
QNLI_label_ten_B[QNLI_label_ten_B==1]=17
#18
QQP_label_ten_B = torch.ones(int(QQP_ten_B.shape[0]),dtype=torch.int64)
QQP_label_ten_B[QQP_label_ten_B==1]=18
#19
WNLI_label_ten_B = torch.ones(int(WNLI_ten_B.shape[0]),dtype=torch.int64)
WNLI_label_ten_B[WNLI_label_ten_B==1]=19
#20
STSB_label_ten_B = torch.ones(int(STSB_ten_B.shape[0]),dtype=torch.int64)
STSB_label_ten_B[STSB_label_ten_B==1]=20
'''
#21
laptop_label_ten_B = torch.ones(int(laptop_ten_B.shape[0]),dtype=torch.int64)
laptop_label_ten_B[laptop_label_ten_B==1]=21
#22
restaurant_label_ten_B = torch.ones(int(restaurant_ten_B.shape[0]),dtype=torch.int64)
restaurant_label_ten_B[restaurant_label_ten_B==1]=22
#23
IMDB_label_ten_B = torch.ones(int(IMDB_ten_B.shape[0]),dtype=torch.int64)
IMDB_label_ten_B[IMDB_label_ten_B==1]=23




#print(sst2_label_ten.shape)
#print(rte_label_ten.shape)
#print(re_label_ten.shape)

#92%
#all_prompt_emb = torch.stack([sst2_ten,rte_ten,re_ten,MNLI_ten,MRPC_ten,QNLI_ten,QQP_ten,WNLI_ten,STSB_ten,laptop_ten,restaurant_ten,IMDB_ten])

#all_prompt_emb = torch.stack([sst2_ten,rte_ten,re_ten,MNLI_ten,MRPC_ten,QNLI_ten,QQP_ten,WNLI_ten,STSB_ten,laptop_ten,restaurant_ten,IMDB_ten,sst2_ten_B,rte_ten_B,re_ten_B,MNLI_ten_B,MRPC_ten_B,QNLI_ten_B,QQP_ten_B,WNLI_ten_B,STSB_ten_B,laptop_ten_B,restaurant_ten_B,IMDB_ten_B])


all_prompt_emb = torch.stack([laptop_ten,restaurant_ten,IMDB_ten,laptop_ten_B,restaurant_ten_B,IMDB_ten_B])

#100%
###
#all_prompt_emb = torch.stack([sst2_ten,re_ten,laptop_ten,restaurant_ten,IMDB_ten])

#all_label = torch.stack([sst2_label_ten,rte_label_ten,re_label_ten,MNLI_label_ten,MRPC_label_ten,QNLI_label_ten,QQP_label_ten,WNLI_label_ten,STSB_label_ten,laptop_label_ten,restaurant_label_ten,IMDB_label_ten,sst2_label_ten_B,rte_label_ten_B,re_label_ten_B,MNLI_label_ten_B,MRPC_label_ten_B,QNLI_label_ten_B,QQP_label_ten_B,WNLI_label_ten_B,STSB_label_ten_B,laptop_label_ten_B,restaurant_label_ten_B,IMDB_label_ten_B])

all_label = torch.stack([laptop_label_ten,restaurant_label_ten,IMDB_label_ten,laptop_label_ten_B,restaurant_label_ten_B,IMDB_label_ten_B])


print("===================")
print("===================")

##3D or 2D
dim=3
#compressed_prompt_emb = train_AE(input=all_prompt_emb,out_features=dim)
#################

compressed_prompt_emb = PCA_svd(X=all_prompt_emb,k=dim)
print(compressed_prompt_emb.shape)

#all: 92%
#sentiment: 100%

#exit()



##################
#######Draw
##################
compressed_prompt_emb = compressed_prompt_emb.to("cpu").detach().numpy()
all_label = all_label.to("cpu").numpy()

#color table: https://www.computerhope.com/htmcolor.htm#color-codes

#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}

#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB",12:"sst2_B",13:"rte_B",14:"re_B",15:"MNLI_B",16:"MRPC_B",17:"QNLI_B",18:"QQP_B",19:"WNLI_B",20:"STSB_B",21:"laptop_B",22:"restaurant_B",23:"IMDB_B"}

task_map={0:"laptop",1:"restaurant",2:"IMDB",3:"laptop_B",4:"restaurant_B",5:"IMDB_B"}

###
#task_map={0:"sst2",1:"re",2:"laptop",3:"restaurant",4:"IMDB"}

#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"QNLI",6:"#CB6D51",7:"#556B2F",8:"STSB",9:"#4863A0",10:"#151B8D"}

#color_map={0:"#186A3B",1:"#1D8348",2:"#239B56",3:"#28B463",4:"#145A32",5:"#196F3D",6:"#1E8449",7:"#229954",8:"#0B5345 ",9:"#0E6655",10:"#117A65",11:"#CD6155",12:"#C0392B",13:"#A93226",14:"#922B21",15:"#7B241C",16:"#943126",17:"#B03A2E",18:"#CB4335",19:"#E74C3C",20:"#EC7063",21:"#F1948A",22:"#D98880",23:"#E6B0AA"}
color_map={0:"#186A3B",1:"#1D8348",2:"#239B56",3:"#F1948A",4:"#D98880",5:"#E6B0AA"}


blocked_list = []
#blocked_list = [1,3,4,5,6,7,8]
#blocked_list = [1,4,7,8]

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
plt.savefig('roberta_bert_prompt.jpg')
#plt.savefig('exp_results/SENTIMENT.jpg')
#plt.savefig('exp_results/NLI.jpg')
#plt.savefig('exp_results/domain.jpg')
#plt.savefig('exp_results/PCA_DOMAIN_BASE_SENTIMENT.jpg')


