import config

import argparse
import os
import logging
import torch
import sys


# [12, 64, 231, 3072] --> 12, 64, 231(1 or 100), 3072

root_dir = "task_activated_neuron"
#root_dir = "10_12_newest_randomseed_base/task_activated_neuron"
#root_dir = "task_activated_neuron/12layer_1prompt"
#root_dir = "task_activated_neuron/task_activated_neuron_label"
#root_dir = "task_activated_neuron/lastlayer_100prompt_Prompt"

dirs = os.listdir(root_dir)
#dirs = ['random']
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", anliPromptRoberta, "snliPromptRoberta", "RTEPromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]
#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "WNLIPromptRoberta", "snliPromptRoberta", "RTEPromptRoberta", "recastfactualityPromptRoberta", "recastmegaveridicalityPromptRoberta","recastnerPromptRoberta", "recastpunsPromptRoberta","recastsentimentPromptRoberta", "recastverbcornerPromptRoberta","ethicscommonsensePromptRoberta","ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "recastnerPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

#order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta"]

###################
###################
###################
###################

#order_list = ['1-laptop_126','1-IMDB_126','1-tweet_126','1-restaurant_326','2-MNLI_326','1-restaurant_126','4-QQP_326','1-SST2_326','1-SST2_86','1-laptop_326','1-restaurant_86','2-QNLI_326','4-QQP_86','1-IMDB_86','2-snli_326','2-MNLI_86','2-snli_126','1-IMDB_326','2-MNLI_126','4-MRPC_326','2-QNLI_126','4-MRPC_86','1-tweet_86','4-QQP_126','1-laptop_86','1-tweet_326','2-QNLI_86','2-snli_86','1-SST2_126','4-MRPC_126']


order_list = ["1-IMDBPromptRoberta", "1-SST2PromptRoberta", "1-laptopPromptRoberta", "1-restaurantPromptRoberta", "1-movierationalesPromptRoberta", "1-tweetevalsentimentPromptRoberta", "2-MNLIPromptRoberta", "2-QNLIPromptRoberta", "2-snliPromptRoberta", "3-ethicsdeontologyPromptRoberta","3-ethicsjusticePromptRoberta","4-QQPPromptRoberta", "4-MRPCPromptRoberta"]

###################
###################
###################
###################


#order_list = ["IMDBPromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "snliPromptRoberta", "MNLIPromptRoberta", "IMDB_base_emotionPromptRoberta", "MNLI_base_nliPromptRoberta", "laptop_base_emotionPromptRoberta", "laptop_base_nliPromptRoberta", "restaurant_base_emotionPromptRoberta", "restaurant_base_nliPromptRoberta", "snli_base_emotionPromptRoberta","snli_base_nliPromptRoberta","RandomPromptRoberta","IMDB_base_nliPromptRoberta","MNLI_base_emotionPromptRoberta"]
#order_list = ["IMDBPromptRoberta","IMDB_base_emotionPromptRoberta","IMDB_base_nliPromptRoberta"]
#order_list = ["IMDBPromptRoberta","IMDB_base_emotionPromptRoberta"]
#order_list = ["IMDBPromptRoberta","laptop_base_emotionPromptRoberta"]
#order_list = ["snliPromptRoberta","IMDB_base_emotionPromptRoberta"]
#order_list = ["IMDBPromptRoberta","MNLI_base_emotionPromptRoberta"]
#order_list = ["restaurantPromptRoberta","restaurant_base_emotionPromptRoberta"]
#order_list = ["snliPromptRoberta","snli_base_nliPromptRoberta"]
#order_list = ["SST2PromptRoberta","SST2_base_emotionPromptRoberta"]
#order_list = ["snliPromptRoberta","MNLIPromptRoberta"]
#order_list = ["MNLIPromptRoberta","MNLI_base_emotionPromptRoberta"]
#order_list = ["snliPromptRoberta","snli_base_emotionPromptRoberta"]
#order_list = ["snliPromptRoberta","snli_base_emotionPromptRoberta"]


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



l=0
k=0
c=0

##Title
#print(end="\t \t")
'''
print(end="\t")
for name in data_name:
    name = name.replace("PromptRoberta","").replace("recast","").replace("ethics","")
    if len(name)>5:
        name = name[:5]
    print(name, end='\t')
print()
'''


for dir_1 in dirs:
    task_type_1 , dir_1 = dir_1.split("-")
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","")
    #print_name = dir_1.replace("PromptRoberta","").replace("urant","").replace("evalsentiment","").replace("rationales","").replace("recast","").replace("ethics","")
    print_name = dir_1.replace("PromptRoberta","").replace("recast","").replace("ethics","")

    if len(print_name)>5:
        print_name = print_name[:5]


    #if "random" != dir_1:
    #    continue

    #print(print_name, end='\t')
    activated_1 = torch.load(root_dir+"/"+dir_1+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
    #print(activated_1.shape)
    #exit()
    ###
    #activated_1 = activated_1[int(sys.argv[1]):int(sys.argv[1])+3,:,:,:]
    #activated_1 = activated_1[9:12,:,:,:]
    ###
    activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

    activated_1[activated_1>0] = float(1)
    activated_1[activated_1<0] = float(0)

    for dir_2 in dirs:
        task_type_2 , dir_2 = dir_2.split("-")

        #if dir_1.split("_")[0] != dir_2.split("_")[0]:
        #    continue

        activated_2 = torch.load(root_dir+"/"+dir_2+"/"+"task_activated_neuron", map_location=lambda storage, loc: storage)
        ###
        #activated_2 = activated_2[int(sys.argv[1]):int(sys.argv[1])+3,:,:,:]
        #activated_2 = activated_2[9:12,:,:,:]
        ###
        activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])

        activated_2[activated_2>0] = float(1)
        activated_2[activated_2<0] = float(0)

        sim = cos(activated_1, activated_2)
        #print("{:.2f}".format(float(sim)), end='\t')

        #sim = torch.dist(activated_1, activated_2, 2)
        #print("{:.2f}".format(float(sim)), end='\t')

        #print()
        #print("act_1", len(activated_1[activated_1==1]))
        #print("act_2", len(activated_2[activated_2==1]))
        #print("======================")
        #print("======================")

        #if dir_1.split("_")[0] == dir_2.split("_")[0]:
        #if dir_1 != dir_2:


        if dir_1 != dir_2:
            if task_type_1 != task_type_2:
                pass
            else:
                continue


            #if dir_1 == dir_2:
            #    continue
            print(dir_1, dir_2)
            #print(sim)
            l+= float(sim)
            c+=1
        #same-type
        # elif

    #print()

#print(l/(13*12))
print(l/c)
print(c)





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


