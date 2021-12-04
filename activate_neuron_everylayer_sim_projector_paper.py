import config

import argparse
import os
import logging
import torch
import sys


# [12, 64, 231, 3072] --> 12, 64, 231(1 or 100), 3072

#root_dir = "task_activated_neuron/12layer_1prompt"
#root_dir = "task_activated_neuron/task_activated_neuron_label"
#root_dir = "10_12_newest_randomseed_base/task_activated_neuron"
#root_dir = "task_activated_neuron/lastlayer_100prompt_Prompt"
root_dir = "task_activated_neuron/"
#root_dir = "task_activated_neuron_ffn/"
#root_dir = "task_activated_neuron_plm/"

dirs = os.listdir(root_dir)
#dirs = ['random']
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", anliPromptRoberta, "snliPromptRoberta", "RTEPromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", "snliPromptRoberta", "RTEPromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", "snliPromptRoberta", "RTEPromptRoberta", "recastfactualityPromptRoberta", "recastmegaveridicalityPromptRoberta","recastnerPromptRoberta", "recastpunsPromptRoberta","recastsentimentPromptRoberta", "recastverbcornerPromptRoberta","ethicscommonsensePromptRoberta","ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "recastnerPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
#order_list = ['laptop_126' 'IMDB_126' 'tweet_126' 'restaurant_326' 'MNLI_326' 'restaurant_126' 'QQP_326' 'SST2_326' 'SST2_86' 'laptop_326' 'restaurant_86' 'QNLI_326' 'QQP_86' 'IMDB_86' 'snli_326' 'MNLI_86' 'snli_126' 'IMDB_326' 'MNLI_126' 'MRPC_326' 'QNLI_126' 'MRPC_86' 'tweet_86' 'QQP_126' 'laptop_86' 'tweet_326' 'QNLI_86' 'snli_86' 'SST2_126' 'MRPC_126']

#order_list = ["IMDBPromptRoberta","laptopPromptRoberta","restaurantPromptRoberta","MNLIPromptRoberta","QNLIPromptRoberta","snliPromptRoberta"]
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta"]

#order_list = ["IMDB","laptop","restaurant","MNLI","QNLI","snli"]
order_list = ["IMDB", "SST2", "laptop", "restaurant", "movierationales", "tweetevalsentiment", "MNLI", "QNLI", "snli"]

#order_list = ["IMDB","SST2","laptop","restaurant","movierationales","tweetevalsentiment","MNLI","QNLI","snli"]




if "_label" in root_dir:
    order_list = [i+str("_label") for i in order_list]

#dirs = [dir for dir in dirs if ".txt" not in dir and "12layer_1prompt" not in dir]
dirs = order_list

data_name = [dir.replace("PromptRoberta","").replace("urant","").replace("evalsentiment","").replace("rationales","").replace("_label","") for dir in dirs]

cos = torch.nn.CosineSimilarity(dim=0)





#if sys.argv[1] == None:
#    sys.argv=1


#sys.stdout = open(root_dir+"/"+str(sys.argv[1])+'.txt', 'w')
#sys.stdout = open(root_dir+"/"+str("11_12")+'.txt', 'w')
#sys.stdout = open(root_dir+"/"+str("1_2")+'.txt', 'w')


#####################################
#####################################
#####################################






##Title
#print(end="\t \t")
print(end="\t")
for name in data_name:
    name = name.replace("PromptRoberta","").replace("recast","").replace("ethics","")
    if len(name)>5:
        name = name[:5]
    print(name, end='\t')
print()



for dir_1 in dirs:
    #print(dir_1, end='\t')
    dir_1 = dir_1+"PromptRoberta"
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","")
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","").replace("evalsentiment","").replace("rationales","")


    print_name = dir_1.replace("PromptRoberta","").replace("recast","").replace("ethics","")
    if len(print_name)>5:
        print_name = print_name[:5]
    #if "random" != dir_1:
    #    continue

    print(print_name, end='\t')
    activated_1 = torch.load(root_dir+dir_1+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
    ###########
    #activated_1 = activated_1[int(sys.argv[1]):int(sys.argv[1])+1,:,:,:]
    #activated_1 = activated_1[9:,:,:,:]
    #activated_1 = activated_1[9:12,:,:,:]
    #print(activated_1.shape)
    #exit()
    #activated_1 = activated_1[11:12,:,:,:]
    #activated_1 = activated_1[0:2,:,:,:]
    #print(activated_1.shape)
    #exit()
    ###########
    activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

    #activated_1[activated_1>0] = float(1)
    #activated_1[activated_1<0] = float(0)

    for dir_2 in dirs:
        #dir_2 = dir_2+"_proj"
        dir_2 = dir_2+"_trainedPromptRoberta"
        #print(dir_2)
        activated_2 = torch.load(root_dir+dir_2+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
        ###########
        #activated_2 = activated_2[9:,:,:,:]
        #activated_2 = activated_2[10:12,:,:,:]
        #activated_2 = activated_2[9:12,:,:,:]
        #activated_2 = activated_2[10:12,:,:,:]
        #activated_2 = activated_2[11:12,:,:,:]
        #activated_2 = activated_2[0:2,:,:,:]
        ###########
        activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])

        #activated_2[activated_2>0] = float(1)
        #activated_2[activated_2<0] = float(0)


        sim = cos(activated_1, activated_2)
        #print("{:.2f}".format(float(sim)),",", end='\t')
        print("{:.2f}".format(float(sim)),",", end='\t')

        #sim = torch.dist(activated_1, activated_2, 2)
        #print("{:.2f}".format(float(sim)), end='\t')

    print()







