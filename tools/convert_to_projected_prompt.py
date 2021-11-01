import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import random
import numpy as np
#from tools.eval_tool_projector import valid, gen_time_str, output_value
#from tools.eval_tool import valid, gen_time_str, output_value
#from tools.init_tool import init_test_dataset, init_formatter
#from reader.reader import init_dataset, init_formatter, init_test_dataset
#import torch.nn as nn
#import torch.optim as optim
from projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer

def EuclideanDistances(task1_emb,task2_emb):
    print(torch.norm(task1_emb-task2_emb, p='fro'))
    print(1/(torch.norm(task1_emb-task2_emb, p='fro')+1))
    print("=====")


def EuclideanDistances_per_token(task1_emb,task2_emb):
    task1_emb = task1_emb.reshape(100,768)
    task2_emb = task2_emb.reshape(100,768)
    sum_euc = 0
    for idx1, v1 in enumerate(task1_emb):
        #print(idx1)
        #print(v1)
        #print(v1.shape)
        #exit()
        for idx2, v2 in enumerate(task2_emb):
            #euc = torch.norm(v1-v2, p='fro')
            euc = torch.norm(v1-v2, p=2)
            #print(euc)
            sum_euc += euc
    print(float((float(sum_euc/100)/100)))
    print(1/(float((float(sum_euc/100)/100))+1))
    print("=====")



cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity_per_token(task1_emb,task2_emb):
    task1_emb = task1_emb.reshape(100,768)
    task2_emb = task2_emb.reshape(100,768)
    sum_c = 0
    #return cos(task1_emb,task2_emb).sum()
    for idx1, v1 in enumerate(task1_emb):
        for idx2, v2 in enumerate(task2_emb):
            c = cos(v1,v2)
            sum_c += c
    print(float(float(sum_c/100)/100))
    print("=====")


#cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    #print(task1_emb.shape)
    #print(task2_emb.shape)
    #exit()
    print(cos(task1_emb,task2_emb))
    #print(torch.cosine_similarity(task1_emb,task2_emb,dim=0))
    print("=====")



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model_AE = AE_1_layer_mutiple_100(dim_0=76800,dim_1=7680,dim_2=76800).to(device)
model_AE.load_state_dict(torch.load("../model/crossPromptRoberta_emotion_100/127_model_cross_0.759.pkl", map_location=lambda storage, loc:storage))
#model_AE.load_state_dict(torch.load("../model/crossPromptRoberta_emotion_100/94_model_cross_0.89.pkl", map_location=lambda storage, loc:storage))

#prompt = torch.load("../task_prompt_emb/IMDBPromptRoberta/task_prompt", map_location=lambda storage, loc:storage)
#prompt = torch.load("../task_prompt_emb/laptopPromptRoberta/task_prompt", map_location=lambda storage, loc:storage)
prompt = torch.load("../task_prompt_emb/restaurantPromptRoberta/task_prompt", map_location=lambda storage, loc:storage)


prompt = prompt.reshape(1,100*768)
#print(prompt.shape)
#print("===========")
#print(prompt)

p_prompt = torch.Tensor(model_AE(prompt))
p_prompt = p_prompt.reshape(76800)
prompt = prompt.reshape(76800)
#print(p_prompt.shape)
#print("===========")
#exit()
#print(p_prompt)
#exit()

EuclideanDistances(prompt, p_prompt)
EuclideanDistances_per_token(prompt, p_prompt)
CosineSimilarity(prompt, p_prompt)
CosineSimilarity_per_token(prompt, p_prompt)
#exit()

p_prompt = p_prompt.reshape(100,768)
print(p_prompt)
print(p_prompt.shape)
#exit()
#torch.save(p_prompt,"IMDBPromptRoberta_proj/task_prompt")
#torch.save(p_prompt,"laptopPromptRoberta_proj/task_prompt")
torch.save(p_prompt,"restaurantPromptRoberta_proj/task_prompt")


