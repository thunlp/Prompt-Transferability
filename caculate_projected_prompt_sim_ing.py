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
from tools.projector import AE_1_layer_mutiple_100, AE_1_layer_mutiple_100_paper
import torch.nn as nn

def EuclideanDistances(task1_emb,task2_emb):
    #print(torch.norm(task1_emb-task2_emb, p='fro'))
    print(1/(torch.norm(task1_emb-task2_emb, p='fro')+1))
    print("=====")


def EuclideanDistances_per_token(task1_emb,task2_emb):
    task1_emb = task1_emb.reshape(100,768)
    task2_emb = task2_emb.reshape(100,768)
    sum_euc = 0
    for idx1, v1 in enumerate(task1_emb):
        for idx2, v2 in enumerate(task2_emb):
            #euc = torch.norm(v1-v2, p='fro')
            euc = torch.norm(v1-v2, p=2)
            sum_euc += euc
    print(1/(float((float(sum_euc/100)/100))+1))
    print("=====")



cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity_per_token(task1_emb,task2_emb):
    task1_emb = task1_emb.reshape(100,768)
    task2_emb = task2_emb.reshape(100,768)
    sum_c = 0
    for idx1, v1 in enumerate(task1_emb):
        for idx2, v2 in enumerate(task2_emb):
            c = cos(v1,v2)
            sum_c += c
    #print(float(float(sum_c/100)/100))
    print((sum_c/float(100))/float(100))
    print("=====")


#cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    print(cos(task1_emb,task2_emb))
    print("=====")


Neuron_cos = torch.nn.CosineSimilarity(dim=0)
def ActivatedNeurons(activated_1, activated_2):
    layer_unit = int(activated_1.shape[0])/3
    print(layer_unit)
    exit()



    #activated_1 = activated_1[:int(sys.argv[1])+1,:,:,:]
    activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

    #activated_2 = activated_2[int(sys.argv[1]):int(sys.argv[1])+1,:,:,:]
    activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])

    #activated_1[activated_1>0] = float(1)
    #activated_1[activated_1<0] = float(0)

    #activated_2[activated_2>0] = float(1)
    #activated_2[activated_2<0] = float(0)

    sim = Neuron_cos(activated_1, activated_2)
    print("{:.2f}".format(float(sim)))





class AE_1_layer_mutiple_100_leaky(nn.Module):
    def __init__(self, **kwargs):
        super(AE_1_layer_mutiple_100_leaky, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["dim_0"], out_features=int(kwargs["dim_1"])
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["dim_1"]), out_features=kwargs["dim_2"]
        )

        self.dim = int(kwargs["dim_2"]/100)

        #########################
        #self.layer_norm = nn.LayerNorm(self.dim, eps=1e-05)
        #########################

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()
        #self.activation = nn.Tanh()
        #self.activation = nn.Softmax(dim=-1)

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = self.activation(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        ###
        #layer_norm = nn.LayerNorm(int(decoded_emb.shape[0]),100,768)
        #decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100,self.dim)
        #decoded_emb = self.layer_norm(decoded_emb)
        #decoded_emb = decoded_emb.reshape(int(decoded_emb.shape[0]),100*self.dim)
        ###
        return decoded_emb





#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


###############
###############

# roberta to roberta Large
#projector = "model/crossPromptRobertaLarge_emotion_100/99_model_cross_0.8539.pkl"
projector = "/home/zhanqimin/prompt_transfer/cross_model_prompt_pair_projector/NLI_T5_leakyrelu/min_loss.pth"
dataset = "MNLI"
base_model = "Roberta"
to = "roberta_to_T5XXL"


'''
# performance bad
projector = "model/crossPromptRobertaLarge_nli_100/14_model_cross_0.6583.pkl"
dataset = "MNLI"
base_model = "Roberta"
to = "roberta_to_robertalarge"
'''




###############
###############


model_parameters = torch.load(projector, map_location=lambda storage, loc:storage)
#print(model_parameters)
for l in model_parameters:
    print(l)
    print(model_parameters[l].shape)
    print("---")
#exit()

#model_AE = AE_1_layer_mutiple_100(dim_0=76800,dim_1=7680,dim_2=76800).to(device)
#model_AE = AE_1_layer_mutiple_100(dim_0=int(model_parameters["encoder.weight"].shape[1]),dim_1=int(model_parameters["encoder.weight"].shape[0]),dim_2=int(model_parameters["decoder.weight"].shape[0])).to(device)
model_AE = AE_1_layer_mutiple_100_leaky(dim_0=int(model_parameters["encoder.weight"].shape[1]),dim_1=int(model_parameters["encoder.weight"].shape[0]),dim_2=int(model_parameters["decoder.weight"].shape[0])).to(device)

model_AE.load_state_dict(model_parameters)

print(model_AE)
exit()



prompt = torch.load("task_prompt_emb/"+str(dataset)+"Prompt"+str(base_model)+"/task_prompt", map_location=lambda storage, loc:storage)



prompt = prompt.reshape(1,int(model_parameters["encoder.weight"].shape[1]))
p_prompt = torch.Tensor(model_AE(prompt))
p_prompt = p_prompt.reshape(100, int(p_prompt.shape[-1]/100))


save_dir = "task_prompt_emb/"+str(dataset)+"Prompt"+str(base_model)+"_"+str(to)
try:
    os.mkdir(save_dir)
except:
    torch.save(p_prompt , save_dir+"/task_prompt")

exit()

###########
###########
###########



p_prompt = p_prompt.reshape(int(model_parameters["encoder.weight"].shape[1]))










prompt = prompt.reshape(int(model_parameters["encoder.weight"].shape[1]))




#######





#EuclideanDistances(prompt, p_prompt)
#EuclideanDistances_per_token(prompt, p_prompt)
#CosineSimilarity(prompt, p_prompt)
#CosineSimilarity_per_token(prompt, p_prompt)
#ActivatedNeurons(prompt_neuron, p_prompt_neuron)

