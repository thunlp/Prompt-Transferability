import config

import argparse
import os
import logging
import torch
import sys


# [12, 64, 231, 3072] --> 12, 64, 231(1 or 100), 3072

root_dir = "task_activated_neuron"
#root_dir = "task_activated_neuron/12layer_1prompt"
#root_dir = "task_activated_neuron/task_activated_neuron_label"
#root_dir = "task_activated_neuron/lastlayer_100prompt_Prompt"

dirs = os.listdir(root_dir)
#dirs = ['random']
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", anliPromptRoberta, "snliPromptRoberta", "RTEPromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", "snliPromptRoberta", "RTEPromptRoberta", "recastfactualityPromptRoberta", "recastmegaveridicalityPromptRoberta","recastnerPromptRoberta", "recastpunsPromptRoberta","recastsentimentPromptRoberta", "recastverbcornerPromptRoberta","ethicscommonsensePromptRoberta","ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "recastnerPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

#order_list = ["IMDBPromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "snliPromptRoberta", "MNLIPromptRoberta", "IMDB_base_emotionPromptRoberta", "MNLI_base_nliPromptRoberta", "laptop_base_emotionPromptRoberta", "laptop_base_nliPromptRoberta", "restaurant_base_emotionPromptRoberta", "restaurant_base_nliPromptRoberta", "snli_base_emotionPromptRoberta","snli_base_nliPromptRoberta","RandomPromptRoberta","IMDB_base_nliPromptRoberta","MNLI_base_emotionPromptRoberta"]
#order_list = ["IMDBPromptRoberta","IMDB_base_emotionPromptRoberta","IMDB_base_nliPromptRoberta"]


if "_label" in root_dir:
    order_list = [i+str("_label") for i in order_list]

#dirs = [dir for dir in dirs if ".txt" not in dir and "12layer_1prompt" not in dir]
dirs = order_list

data_name = [dir.replace("PromptRoberta","").replace("_label","").replace("urant","").replace("evalsentiment","").replace("rationales","") for dir in dirs]

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
    print_name = dir_1.replace("PromptRoberta","").replace("recast","").replace("ethics","")

    if len(print_name)>5:
        print_name = print_name[:5]


    #if "random" != dir_1:
    #    continue

    print(print_name, end='\t')
    activated_1 = torch.load(root_dir+"/"+dir_1+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
    activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

    activated_1[activated_1>0] = float(1)
    activated_1[activated_1<0] = float(0)

    for dir_2 in dirs:
        activated_2 = torch.load(root_dir+"/"+dir_2+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
        activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])

        activated_2[activated_2>0] = float(1)
        activated_2[activated_2<0] = float(0)

        sim = cos(activated_1, activated_2)
        print("{:.2f}".format(float(sim)), end='\t')

        #sim = torch.dist(activated_1, activated_2, 2)
        #print("{:.2f}".format(float(sim)), end='\t')

    print()







'''
#Check by each sentence
##Title
#print(end="\t \t")
print(end="\t")
for name in data_name:
    print(name, end='\t')
print()

for dir_1 in dirs:
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","")
    print_name = dir_1.replace("PromptRoberta","").replace("urant","").replace("evalsentiment","").replace("rationales","")

    #if "random" != dir_1:
    #    continue

    print(print_name, end='\t')
    activated_1 = torch.load(root_dir+"/"+dir_1+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
    #activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
    activated_1 = activated_1.reshape(activated_1.shape[1],activated_1.shape[0]*activated_1.shape[2]*activated_1.shape[3])

    activated_1[activated_1>0] = float(1)
    activated_1[activated_1<0] = float(0)

    for dir_2 in dirs:
        activated_2 = torch.load(root_dir+"/"+dir_2+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
        #activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        activated_2 = activated_2.reshape(activated_2.shape[1],activated_2.shape[0]*activated_2.shape[2]*activated_2.shape[3])

        activated_2[activated_2>0] = float(1)
        activated_2[activated_2<0] = float(0)

        sim = 0
        for number_sentence in range(int(activated_1.shape[0])):
            sim += cos(activated_1[number_sentence], activated_2[number_sentence])
        sim = sim/int(activated_1.shape[0])
        print("{:.2f}".format(float(sim)), end='\t')

        #sim = torch.dist(activated_1, activated_2, 2)
        #print("{:.2f}".format(float(sim)), end='\t')

    print()
'''











'''
topk=50
#topk= 12*64*231*3072 - 1

print("===========================================")
print("===============Top",topk,"================")
print("===========================================")


##Title
#print(end="\t \t")
print(end="\t")
for name in data_name:
    print(name, end='\t')
print()


def matching(activated_1,activated_2,topk):
    activated_1 = [int(a) for a in activated_1]
    activated_2 = [int(a) for a in activated_2]
    return float(len(set(activated_1) & set(activated_2))/topk)


for dir_1 in dirs:
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","")
    print_name = dir_1.replace("PromptRoberta","").replace("urant","").replace("evalsentiment","").replace("rationales","")

    #if "random" != dir_1:
    #    continue

    print(print_name, end='\t')
    activated_1 = torch.load(root_dir+"/"+dir_1+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
    activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

    #activated_1[activated_1>0] = float(1)
    #activated_1[activated_1<0] = float(0)

    activated_1 = torch.topk(activated_1, topk).indices

    for dir_2 in dirs:
        activated_2 = torch.load(root_dir+"/"+dir_2+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
        activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])

        #activated_2[activated_2>0] = float(1)
        #activated_2[activated_2<0] = float(0)

        activated_2 = torch.topk(activated_2, topk).indices

        #sim = cos(activated_1, activated_2)
        sim = matching(activated_1,activated_2,topk)
        print("{:.2f}".format(float(sim)), end='\t')

        #sim = torch.dist(activated_1, activated_2, 2)
        #print("{:.2f}".format(float(sim)), end='\t')

    print()
'''




'''
#Check by each layer
#topk=12*1*3072
#topk=12*1*3072
#topk= 12*64*1*3072 - 1
#topk=100

print("===========================================")
print("===============Top",topk,"================")
print("===========================================")


##Title
#print(end="\t \t")
print(end="\t")
for name in data_name:
    print(name, end='\t')
print()


def matching(activated_1,activated_2,topk):
    activated_1 = [int(a) for a in activated_1]
    activated_2 = [int(a) for a in activated_2]
    return float(len(set(activated_1) & set(activated_2))/topk)


for dir_1 in dirs:
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","")
    print_name = dir_1.replace("PromptRoberta","").replace("urant","").replace("evalsentiment","").replace("rationales","")

    #if "random" != dir_1:
    #    continue

    print(print_name, end='\t')
    activated_1 = torch.load(root_dir+"/"+dir_1+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
    #activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
    #activated_1 = activated_1.reshape(activated_1.shape[1],activated_1.shape[0]*activated_1.shape[2]*activated_1.shape[3])
    activated_1 = activated_1.reshape(activated_1.shape[0],activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])


    #activated_1[activated_1>0] = float(1)
    #activated_1[activated_1<0] = float(0)

    #activated_1 = torch.topk(activated_1, topk).indices

    for dir_2 in dirs:
        activated_2 = torch.load(root_dir+"/"+dir_2+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
        #activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        #activated_2 = activated_2.reshape(activated_2.shape[1],activated_2.shape[0]*activated_2.shape[2]*activated_2.shape[3])
        activated_2 = activated_2.reshape(activated_2.shape[0],activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])

        #activated_2[activated_2>0] = float(1)
        #activated_2[activated_2<0] = float(0)

        #activated_2 = torch.topk(activated_2, topk).indices


        sim = 0
        for number_layer in range(int(activated_1.shape[0])):
        #for number_layer in [9,10,11]:
            sim += matching(activated_1[number_layer].topk(topk).indices, activated_2[number_layer].topk(topk).indices, topk)
        sim = sim/int(activated_1.shape[0])
        #sim = sim/int(3)
        print("{:.2f}".format(float(sim)), end='\t')

        #sim = torch.dist(activated_1, activated_2, 2)
        #print("{:.2f}".format(float(sim)), end='\t')

    print()
'''


