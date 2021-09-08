import config

import argparse
import os
import logging
import torch

root_dir = "task_activated_neuron"
dirs = os.listdir(root_dir)
#dirs = ['random']
dirs = [dir for dir in dirs if ".txt" not in dir and "12layer_1prompt" not in dir]
print(dirs)
#exit()
data_name = [dir.replace("PromptRoberta","").replace("urant","") for dir in dirs]


cos = torch.nn.CosineSimilarity(dim=0)


'''
##Title
#print(end="\t \t")
print(end="\t")
for name in data_name:
    print(name, end='\t')
print()

for dir_1 in dirs:
    print_name = dir_1.replace("PromptRoberta","").replace("urant","")

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




topk=1000

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
    print_name = dir_1.replace("PromptRoberta","").replace("urant","")

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

        '''
        sim = torch.dist(activated_1, activated_2, 2)
        print("{:.2f}".format(float(sim)), end='\t')
        '''

    print()
