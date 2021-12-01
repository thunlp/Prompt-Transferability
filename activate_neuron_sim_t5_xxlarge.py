import config

import argparse
import os
import logging
import torch
import sys


# [12, 64, 231, 3072] --> 12, 64, 231(1 or 100), 3072

#root_dir = "task_activated_neuron_xxlarge"
root_dir = "task_activated_neuron_xxlarge/neurons/final"
#root_dir = "task_activated_neuron_xxlarge/neurons/wi0"
#root_dir = "task_activated_neuron_xxlarge/neurons/wi1"
#root_dir = "task_activated_neuron_xxlarge/neurons/final"
#root_dir = "task_activated_neuron"
#root_dir = "task_activated_neuron_ffn"
#root_dir = "task_activated_neuron/12layer_1prompt"
#root_dir = "task_activated_neuron/task_activated_neuron_label"
#root_dir = "task_activated_neuron/lastlayer_100prompt_Prompt"

dirs = os.listdir(root_dir)
#dirs = ['random']
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", anliPromptRoberta, "snliPromptRoberta", "RTEPromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", "snliPromptRoberta", "RTEPromptRoberta", "recastfactualityPromptRoberta", "recastmegaveridicalityPromptRoberta","recastnerPromptRoberta", "recastpunsPromptRoberta","recastsentimentPromptRoberta", "recastverbcornerPromptRoberta","ethicscommonsensePromptRoberta","ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "recastnerPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]


#order_list = ["IMDBPromptT5", "SST2PromptT5", "laptopPromptT5", "restaurantPromptT5", "movierationalesPromptT5", "tweetevalsentimentPromptT5", "MNLIPromptT5", "QNLIPromptT5", "snliPromptT5", "ethicsdeontologyPromptT5","ethicsjusticePromptT5","QQPPromptT5", "MRPCPromptT5","squadPromptT5","nq_openPromptT5","samsumPromptT5","multi_newsPromptT5"]

#order_list = ["IMDB", "sst-2", "laptop", "restaurant", "movie", "tweet", "MNLI", "QNLI", "snli", "deont","justice","QQP", "MRPC","squad","nqopen","samsum","multinews"]
order_list = ["IMDB", "sst-2", "laptop", "restaurant", "movie", "tweet", "MNLI", "QNLI", "snli", "deont","justice","QQP", "MRPC","squad","nqopen","multinews"]


#order_list = ["IMDBPromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "snliPromptRoberta", "MNLIPromptRoberta", "IMDB_base_emotionPromptRoberta", "MNLI_base_nliPromptRoberta", "laptop_base_emotionPromptRoberta", "laptop_base_nliPromptRoberta", "restaurant_base_emotionPromptRoberta", "restaurant_base_nliPromptRoberta", "snli_base_emotionPromptRoberta","snli_base_nliPromptRoberta","RandomPromptRoberta","IMDB_base_nliPromptRoberta","MNLI_base_emotionPromptRoberta"]
#order_list = ["IMDBPromptRoberta","IMDB_base_emotionPromptRoberta","IMDB_base_nliPromptRoberta"]


if "_label" in root_dir:
    order_list = [i+str("_label") for i in order_list]

#dirs = [dir for dir in dirs if ".txt" not in dir and "12layer_1prompt" not in dir]
dirs = order_list

data_name = [dir.replace("PromptT5","").replace("_label","").replace("urant","").replace("evalsentiment","").replace("rationales","") for dir in dirs]

cos = torch.nn.CosineSimilarity(dim=0)







#topk=100
topk=100


#sys.stdout = open(root_dir+"/"+str(sys.argv[1])+'.txt', 'w')
#sys.stdout = open(root_dir+"/"+'12layers_1st.txt', 'w')
#sys.stdout = open(root_dir+"/"+'all_12layers_1st.txt', 'w')
#sys.stdout = open(root_dir+"/"+"12layers_1st_top"+str(topk)+".txt", 'w')
#sys.stdout = open(root_dir+"/"+"12layers_1st_top"+str(topk)+".txt", 'w')


#####################################
#####################################
#####################################


#threadhold=0.01
threadhold=0.0

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
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","")
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","").replace("evalsentiment","").replace("rationales","").replace("recast","").replace("ethics","")
    print_name = dir_1.replace("PromptT5","").replace("recast","").replace("ethics","")

    if len(print_name)>5:
        print_name = print_name[:5]


    #if "random" != dir_1:
    #    continue

    print(print_name, end='\t')
    activated_1 = torch.load(root_dir+"/"+dir_1+"/"+"neurons.pt", map_location=lambda storage, loc: storage)
    #print(activated_1.shape)
    #exit()
    #####
    #activated_1 = activated_1[:,1:2,:,:]
    #print(activated_1.shape)
    #exit()
    activated_1 = activated_1.reshape(24,10240)
    activated_1 = activated_1[18:24,:]
    #####
    #activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
    activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])

    activated_1[activated_1>threadhold] = float(1)
    activated_1[activated_1<threadhold] = float(0)

    for dir_2 in dirs:
        activated_2 = torch.load(root_dir+"/"+dir_2+"/"+"neurons.pt", map_location=lambda storage, loc: storage)
        #####
        #activated_2 = activated_2[:,1:2,:,:]
        activated_2 = activated_2.reshape(24,10240)
        activated_2 = activated_2[18:24,:]
        #####
        #activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])

        activated_2[activated_2>threadhold] = float(1)
        activated_2[activated_2<threadhold] = float(0)


        sim = cos(activated_1.float(), activated_2.float())
        print("{:.2f}".format(float(sim)), end='\t')

        #sim = torch.dist(activated_1, activated_2, 2)
        #print("{:.2f}".format(float(sim)), end='\t')

    print()







