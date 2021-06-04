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
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/"
sst2_file = os.listdir(path)
print(sst2_file)
for file in sst2_file:
    #print(torch.load(path+file).shape)
    sst2_ten.append(torch.load(path+file))
sst2_ten = torch.cat(sst2_ten)
print(sst2_ten.shape)
sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]),int(sst2_ten.shape[1]*sst2_ten.shape[2]))
print(sst2_ten.shape)
print(sst2_ten)
exit()
#exit()

#RTE
rte_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/RTEPromptRoberta/"
rte_file = os.listdir(path)
print(rte_file)
for file in rte_file:
    #print(torch.load(path+file).shape)
    rte_ten.append(torch.load(path+file))
rte_ten = torch.cat(rte_ten)
print(rte_ten.shape)
rte_ten = rte_ten.reshape(int(rte_ten.shape[0]),int(rte_ten.shape[1]*rte_ten.shape[2]))
print(rte_ten.shape)
#exit()


#RE
re_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/REPrompt/"
re_file = os.listdir(path)
print(re_file)
for file in re_file:
    #print(torch.load(path+file).shape)
    re_ten.append(torch.load(path+file))
re_ten = torch.cat(re_ten)
print(re_ten.shape)
re_ten = re_ten.reshape(int(re_ten.shape[0]),int(re_ten.shape[1]*re_ten.shape[2]))
print(re_ten.shape)
#exit()


###########################
###########################
###########################

task_map={0:"sst2",1:"rte",2:"re"}

#0
sst2_label_ten = torch.zeros(int(sst2_ten.shape[0]),dtype=torch.int32)

#1
rte_label_ten = torch.ones(int(rte_ten.shape[0]),dtype=torch.int32)

#1
re_label_ten = torch.ones(int(re_ten.shape[0]),dtype=torch.int32)
re_label_ten[re_label_ten==1]=2

print(sst2_label_ten.shape)
print(rte_label_ten.shape)
print(re_label_ten.shape)

all_prompt_emb = torch.cat([sst2_ten,rte_ten,re_ten]).to("cpu").numpy()
all_label = torch.cat([sst2_label_ten,rte_label_ten,re_label_ten]).to("cpu").numpy()


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


