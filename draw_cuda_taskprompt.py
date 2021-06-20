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
from openTSNE_.examples import utils_
#import utils
import numpy as np
import matplotlib.pyplot as plt

from tsnecuda import TSNE
#from os import listdir
#from os.path import isfile, join
import glob

#####


def plot(x, y, **kwargs):
    utils_.plot(
        x,
        y,
        colors=utils_.MOUSE_10X_COLORS,
        alpha=kwargs.pop("alpha", 0.1),
        draw_legend=True,
        draw_centers=True,
        label_map=kwargs["label_map"]
        **kwargs,
    )





#SST2
sst2_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/task_prompt"
sst2_ten = torch.load(path).view(76800)
print(sst2_ten.shape)
sst2_ten = torch.stack([sst2_ten for i in range(200)])
print(sst2_ten.shape)


#laptop
laptop_ten = list()
laptop_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/laptopPromptRoberta/task_prompt"
laptop_ten = torch.load(path).view(76800)
print(laptop_ten.shape)
laptop_ten = torch.stack([laptop_ten for i in range(200)])
print(laptop_ten.shape)


#restaurant
restaurant_ten = list()
restaurant_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/restaurantPromptRoberta/task_prompt"
restaurant_ten = torch.load(path).view(76800)
print(restaurant_ten.shape)
restaurant_ten = torch.stack([restaurant_ten for i in range(200)])
print(restaurant_ten.shape)

#############
#############



#RTE
rte_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/RTEPromptRoberta/task_prompt"
rte_ten = torch.load(path).view(76800)
print(rte_ten.shape)
rte_ten = torch.stack([rte_ten for i in range(200)])
print(rte_ten.shape)


#MNLI
MNLI_ten = list()
MNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MNLIPromptRoberta/task_prompt"
MNLI_ten = torch.load(path).view(76800)
print(MNLI_ten.shape)
MNLI_ten = torch.stack([MNLI_ten for i in range(200)])
print(MNLI_ten.shape)



#WNLI
WNLI_ten = list()
WNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/WNLIPromptRoberta/task_prompt"
WNLI_ten = torch.load(path).view(76800)
print(WNLI_ten.shape)
WNLI_ten = torch.stack([WNLI_ten for i in range(200)])
print(WNLI_ten.shape)


################
#Paraphrase
################

#MRPC
MRPC_ten = list()
MRPC_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MRPCPromptRoberta/task_prompt"
MRPC_ten = torch.load(path).view(76800)
print(MRPC_ten.shape)
MRPC_ten = torch.stack([MRPC_ten for i in range(200)])
print(MRPC_ten.shape)


#QQP
QQP_ten = list()
QQP_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QQPPromptRoberta/task_prompt"
QQP_ten = torch.load(path).view(76800)
print(QQP_ten.shape)
QQP_ten = torch.stack([QQP_ten for i in range(200)])
print(QQP_ten.shape)


################
#RE
################

#RE
re_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/REPrompt/task_prompt"
re_ten = torch.load(path).view(76800)
print(re_ten.shape)
re_ten = torch.stack([re_ten for i in range(200)])
print(re_ten.shape)


################
#Other
################


#QNLI
QNLI_ten = list()
QNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QNLIPromptRoberta/task_prompt"
QNLI_ten = torch.load(path).view(76800)
print(QNLI_ten.shape)
QNLI_ten = torch.stack([QNLI_ten for i in range(200)])
print(QNLI_ten.shape)



#STSB
STSB_ten = list()
STSB_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/STSBPromptRoberta/task_prompt"
STSB_ten = torch.load(path).view(76800)
print(STSB_ten.shape)
STSB_ten = torch.stack([STSB_ten for i in range(200)])
print(STSB_ten.shape)



###########################
###########################
###########################

task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant"}

#task_map={0:"sst2",1:"rte",2:"re"}

#0
sst2_label_ten = torch.zeros(int(sst2_ten.shape[0]),dtype=torch.int32)
#1
rte_label_ten = torch.ones(int(rte_ten.shape[0]),dtype=torch.int32)
#2
re_label_ten = torch.ones(int(re_ten.shape[0]),dtype=torch.int32)
re_label_ten[re_label_ten==1]=2
#3
MNLI_label_ten = torch.ones(int(MNLI_ten.shape[0]),dtype=torch.int32)
MNLI_label_ten[MNLI_label_ten==1]=3
#4
MRPC_label_ten = torch.ones(int(MRPC_ten.shape[0]),dtype=torch.int32)
MRPC_label_ten[MRPC_label_ten==1]=4
#5
QNLI_label_ten = torch.ones(int(QNLI_ten.shape[0]),dtype=torch.int32)
QNLI_label_ten[QNLI_label_ten==1]=5
#6
QQP_label_ten = torch.ones(int(QQP_ten.shape[0]),dtype=torch.int32)
QQP_label_ten[QQP_label_ten==1]=6
#7
WNLI_label_ten = torch.ones(int(WNLI_ten.shape[0]),dtype=torch.int32)
WNLI_label_ten[WNLI_label_ten==1]=7
#8
STSB_label_ten = torch.ones(int(STSB_ten.shape[0]),dtype=torch.int32)
STSB_label_ten[STSB_label_ten==1]=8
#9
laptop_label_ten = torch.ones(int(laptop_ten.shape[0]),dtype=torch.int32)
laptop_label_ten[laptop_label_ten==1]=9
#10
restaurant_label_ten = torch.ones(int(restaurant_ten.shape[0]),dtype=torch.int32)
restaurant_label_ten[restaurant_label_ten==1]=10

#print(sst2_label_ten.shape)
#print(rte_label_ten.shape)
#print(re_label_ten.shape)

all_prompt_emb = torch.cat([sst2_ten,rte_ten,re_ten,MNLI_ten,MRPC_ten,QNLI_ten,QQP_ten,WNLI_ten,STSB_ten,laptop_ten,restaurant_ten]).to("cpu").numpy()
all_label = torch.cat([sst2_label_ten,rte_label_ten,re_label_ten,MNLI_label_ten,MRPC_label_ten,QNLI_label_ten,QQP_label_ten,WNLI_label_ten,STSB_label_ten,laptop_label_ten,restaurant_label_ten]).to("cpu").numpy()

#print(all_prompt_emb.shape)
#print(all_label.shape)
#exit()


#1200 --> 2400 --> 50

tsne = TSNE(
    perplexity=32,
    n_iter=1000,
    metric="euclidean",
    init='random',
    n_components=2,
    random_seed=42,
    device=0,
)

'''
tsne = TSNE(
    perplexity=64,
    n_iter=1200,
    metric="euclidean",
    callbacks=ErrorLogger(),
    n_jobs=64,
    random_state=42,
    learning_rate='auto',
    initialization='pca',
    n_components=2,
)
'''

#sst2_ten = sst2_ten.to("cpu").numpy()

print(all_prompt_emb.shape)

embedding_train = tsne.fit_transform(all_prompt_emb)
#utils_.plot(x=embedding_train, y=all_label, colors=utils_.MOUSE_10X_COLORS, label_map=task_map)
utils_.plot(x=embedding_train, y=all_label, colors=utils_.MOUSE_10X_COLORS, label_map=task_map)



plt.title("Task Prompt Dist")
plt.savefig('output.pdf')


