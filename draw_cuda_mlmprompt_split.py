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


from openTSNE_.examples import utils_
import numpy as np
import matplotlib.pyplot as plt

from tsnecuda import TSNE
import glob

#####




def PCA_svd(X=None, k=None, center=True):
    #############original##############
    X = X.to("cpu")
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



#SST2
sst2_s1_ten = list()
path="task_prompt_emb/SST2PromptRoberta_mlm_s1/task_prompt"
sst2_s1_ten = torch.load(path).view(76800).to("cpu")
print(sst2_s1_ten.shape)

sst2_s2_ten = list()
path="task_prompt_emb/SST2PromptRoberta_mlm_s2/task_prompt"
sst2_s2_ten = torch.load(path).view(76800).to("cpu")
print(sst2_s2_ten.shape)


#agnews
agnews_s1_ten = list()
path="task_prompt_emb/agnewsPromptRoberta_mlm_s1/task_prompt"
agnews_s1_ten = torch.load(path).view(76800).to("cpu")
print(agnews_s1_ten.shape)

agnews_s2_ten = list()
path="task_prompt_emb/agnewsPromptRoberta_mlm_s2/task_prompt"
agnews_s2_ten = torch.load(path).view(76800).to("cpu")
print(agnews_s2_ten.shape)



#cs_wiki
cs_wiki_s1_ten = list()
path="task_prompt_emb/cs_wikiPromptRoberta_mlm_s1/task_prompt"
cs_wiki_s1_ten = torch.load(path).view(76800).to("cpu")
print(cs_wiki_s1_ten.shape)

cs_wiki_s2_ten = list()
path="task_prompt_emb/cs_wikiPromptRoberta_mlm_s2/task_prompt"
cs_wiki_s2_ten = torch.load(path).view(76800).to("cpu")
print(cs_wiki_s2_ten.shape)



#scierc
scierc_s1_ten = list()
path="task_prompt_emb/sciercPromptRoberta_mlm_s1/task_prompt"
scierc_s1_ten = torch.load(path).view(76800).to("cpu")
print(scierc_s1_ten.shape)

scierc_s2_ten = list()
path="task_prompt_emb/sciercPromptRoberta_mlm_s2/task_prompt"
scierc_s2_ten = torch.load(path).view(76800).to("cpu")
print(scierc_s2_ten.shape)



#imdb
imdb_s1_ten = list()
path="task_prompt_emb/IMDBPromptRoberta_mlm_s1/task_prompt"
imdb_s1_ten = torch.load(path).view(76800).to("cpu")

imdb_s2_ten = list()
path="task_prompt_emb/IMDBPromptRoberta_mlm_s2/task_prompt"
imdb_s2_ten = torch.load(path).view(76800).to("cpu")

###########################
###########################
###########################

#task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}

#task_map={0:"restaurant",1:"laptop",2:"IMDB",3:"SST2",4:"MRPC"}
task_map={0:"sst2_s1",1:"sst2_s1",2:"agnews_s1",3:"agnews_s2",4:"scierc_s1",5:"scierc_s2",6:"cs_wiki_s1",7:"cs_wiki_s2",8:"imdb_s1",9:"imdb_s2"}


#92%
#all_prompt_emb = torch.stack([sst2_ten,rte_ten,re_ten,MNLI_ten,MRPC_ten,QNLI_ten,QQP_ten,WNLI_ten,STSB_ten,laptop_ten,restaurant_ten,IMDB_ten])
#all_prompt_emb = torch.stack([restaurant_ten, laptop_ten, IMDB_ten, sst2_ten, MRPC_ten])
all_prompt_emb = torch.stack([sst2_s1_ten, sst2_s2_ten, agnews_s1_ten, agnews_s2_ten, scierc_s1_ten, scierc_s2_ten, cs_wiki_s1_ten, cs_wiki_s2_ten, imdb_s1_ten, imdb_s2_ten])
#all_prompt_emb = torch.stack([restaurant_ten, laptop_ten, IMDB_ten, sst2_ten])

#100%

###
#all_prompt_emb = torch.stack([sst2_ten,re_ten,laptop_ten,restaurant_ten,IMDB_ten])


#all_label = torch.stack([sst2_label_ten,rte_label_ten,re_label_ten,MNLI_label_ten,MRPC_label_ten,QNLI_label_ten,QQP_label_ten,WNLI_label_ten,STSB_label_ten,laptop_label_ten,restaurant_label_ten,IMDB_label_ten])


print("===================")
print("===================")

##3D or 2D
dim=3


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
###
#task_map={0:"sst2",1:"re",2:"laptop",3:"restaurant",4:"IMDB"}

#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"QNLI",6:"#CB6D51",7:"#556B2F",8:"STSB",9:"#4863A0",10:"#151B8D"}
#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"#64CD64",6:"#CB6D51",7:"#556B2F",8:"#FFC0CB",9:"#4863A0",10:"#151B8D",11:"#00FFFF"}
color_map={0:"#728FCE",1:"#728FCE",2:"#3D0C02",3:"#3D0C02",4:"#C04000",5:"#C04000",6:"#CB6D51",7:"#CB6D51",8:"#FFC0CB",9:"#FFC0CB"}


blocked_list = []
#blocked_list = [4]
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



#plt.legend()
#plt.legend(loc="upper left")
#plt.title("Task Prompt Dist")
#plt.savefig('output.pdf')
plt.savefig('output.jpg')
#plt.savefig('exp_results/SENTIMENT.jpg')
#plt.savefig('exp_results/NLI.jpg')
#plt.savefig('exp_results/domain.jpg')
#plt.savefig('exp_results/PCA_DOMAIN_BASE_SENTIMENT.jpg')


