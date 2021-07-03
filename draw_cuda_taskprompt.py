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

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["input_shape"], out_features=kwargs["out_features"]
        )
        '''
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=76800
        )
        self.encoder_output_layer = nn.Linear(
            in_features=76800, out_features=2
        )
        '''
        self.decoder = nn.Linear(
            in_features=kwargs["out_features"], out_features=kwargs["input_shape"]
        )
        '''
        self.decoder_hidden_layer = nn.Linear(
            in_features=2, out_features=76800
        )
        self.decoder_output_layer = nn.Linear(
            in_features=76800, out_features=kwargs["input_shape"]
        )
        '''

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        '''
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
        '''
        encoded_emb = self.encoding(features)
        decoded_emb = self.decoding(encoded_emb)
        return decoded_emb



def train_AE(input=None, out_features=None):

    ##################
    #######AE training
    ##################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(input_shape=int(input.shape[-1]),out_features=out_features).to(device)

    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # mean-squared error loss
    criterion = nn.MSELoss()

    epochs=10
    iterations=100
    model.train()
    for epoch in range(epochs):
        loss = 0
        #for batch_features, _ in train_loader:
        for iter in range(iterations):
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = all_prompt_emb.view(-1, int(input.shape[-1])).to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        #loss = loss / len(train_loader)
        loss = loss / iterations

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    ##################
    #######AE Inference
    ##################
    model.eval().to('cpu')
    compressed_prompt_emb = model.encoder(all_prompt_emb)
    print(compressed_prompt_emb.shape)
    return compressed_prompt_emb



def PCA_svd(X=None, k=None, center=True):
    #############original##############
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
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/IMDBPromptRoberta/task_prompt"
IMDB_ten = torch.load(path).view(76800)
print(IMDB_ten.shape)
#IMDB_ten = torch.stack([IMDB_ten for i in range(200)])
#print(IMDB_ten.shape)


#SST2
sst2_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/task_prompt"
sst2_ten = torch.load(path).view(76800)
print(sst2_ten.shape)
#sst2_ten = torch.stack([sst2_ten for i in range(200)])
#print(sst2_ten.shape)


#laptop
laptop_ten = list()
laptop_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/laptopPromptRoberta/task_prompt"
laptop_ten = torch.load(path).view(76800)
print(laptop_ten.shape)
#laptop_ten = torch.stack([laptop_ten for i in range(200)])
#print(laptop_ten.shape)


#restaurant
restaurant_ten = list()
restaurant_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/restaurantPromptRoberta/task_prompt"
restaurant_ten = torch.load(path).view(76800)
print(restaurant_ten.shape)
#restaurant_ten = torch.stack([restaurant_ten for i in range(200)])
#print(restaurant_ten.shape)

#############
#############


#RTE
rte_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/RTEPromptRoberta/task_prompt"
rte_ten = torch.load(path).view(76800)
print(rte_ten.shape)
#rte_ten = torch.stack([rte_ten for i in range(200)])
#print(rte_ten.shape)


#MNLI
MNLI_ten = list()
MNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MNLIPromptRoberta/task_prompt"
MNLI_ten = torch.load(path).view(76800)
print(MNLI_ten.shape)
#MNLI_ten = torch.stack([MNLI_ten for i in range(200)])
#print(MNLI_ten.shape)



#WNLI
WNLI_ten = list()
WNLI_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/WNLIPromptRoberta/task_prompt"
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
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MRPCPromptRoberta/task_prompt"
MRPC_ten = torch.load(path).view(76800)
print(MRPC_ten.shape)
#MRPC_ten = torch.stack([MRPC_ten for i in range(200)])
#print(MRPC_ten.shape)


#QQP
QQP_ten = list()
QQP_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QQPPromptRoberta/task_prompt"
QQP_ten = torch.load(path).view(76800)
print(QQP_ten.shape)
#QQP_ten = torch.stack([QQP_ten for i in range(200)])
#print(QQP_ten.shape)


################
#RE
################
#RE
re_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/REPrompt/task_prompt"
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
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QNLIPromptRoberta/task_prompt"
QNLI_ten = torch.load(path).view(76800)
print(QNLI_ten.shape)
#QNLI_ten = torch.stack([QNLI_ten for i in range(200)])
#print(QNLI_ten.shape)



#STSB
STSB_ten = list()
STSB_ten = list()
path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/STSBPromptRoberta/task_prompt"
STSB_ten = torch.load(path).view(76800)
print(STSB_ten.shape)
#STSB_ten = torch.stack([STSB_ten for i in range(200)])
#print(STSB_ten.shape)



###########################
###########################
###########################

task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}

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
#11
IMDB_label_ten = torch.ones(int(IMDB_ten.shape[0]),dtype=torch.int32)
IMDB_label_ten[IMDB_label_ten==1]=11

#print(sst2_label_ten.shape)
#print(rte_label_ten.shape)
#print(re_label_ten.shape)

#92%
all_prompt_emb = torch.stack([sst2_ten,rte_ten,re_ten,MNLI_ten,MRPC_ten,QNLI_ten,QQP_ten,WNLI_ten,STSB_ten,laptop_ten,restaurant_ten,IMDB_ten])
#100%
all_prompt_emb = torch.stack([sst2_ten,re_ten,laptop_ten,restaurant_ten,IMDB_ten])


all_label = torch.stack([sst2_label_ten,rte_label_ten,re_label_ten,MNLI_label_ten,MRPC_label_ten,QNLI_label_ten,QQP_label_ten,WNLI_label_ten,STSB_label_ten,laptop_label_ten,restaurant_label_ten,IMDB_label_ten])


print("===================")
print("===================")

##3D or 2D
dim=3
#compressed_prompt_emb = train_AE(input=all_prompt_emb,out_features=dim)
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

task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"laptop",10:"restaurant",11:"IMDB"}

#color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"QNLI",6:"#CB6D51",7:"#556B2F",8:"STSB",9:"#4863A0",10:"#151B8D"}
color_map={0:"#728FCE",1:"#347235",2:"#3D0C02",3:"#6B8E23",4:"#C04000",5:"#32CD32",6:"#CB6D51",7:"#556B2F",8:"#FFC0CB",9:"#4863A0",10:"#151B8D",11:"#00FFFF"}


blocked_list = [0,2,9,10,11]
#blocked_list = []

#sentiment, NLI, RE, Paraphrase
#blocked_list = [5,8]


#========================
#Domain

#Wiki, reataurant, computer, movie
#blocked_list = [3,4,6,7]


#========================
#Task

#NLI, RE, sentiment
#blocked_list = [4,5,6,8]

#NLI, RE
#blocked_list = [0,4,5,6,8,9,10]



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
    try:
        print(compressed_prompt_emb[task_id])
    except:
        continue

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
#plt.savefig('exp_results/PCA_DOMAIN_BASE_SENTIMENT.jpg')


