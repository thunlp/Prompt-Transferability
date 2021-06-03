import argparse
import logging
import random
import numpy as np
import os
import json
import math
import numpy

import torch


from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from openTSNE import initialization
from openTSNE.callbacks import ErrorLogger
#from examples import utils
from openTSNE_.examples import utils_
#import utils
import numpy as np
import matplotlib.pyplot as plt

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
        draw_legend=False,
        **kwargs,
    )


def rotate(degrees):
    phi = degrees * np.pi / 180
    return np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)],
    ])



#SST2
sst2_ten = list()
path="/mnt/datadisk0/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/"
sst2_file = os.listdir(path)
print(sst2_file)
for file in sst2_file:
    print(torch.load(path+file).shape)
    sst2_ten.append(torch.load(path+file))
sst2_ten = torch.cat(sst2_ten)
print(sst2_ten.shape)
sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]),int(sst2_ten.shape[1]*sst2_ten.shape[2]))
print(sst2_ten.shape)
#exit()

#RTE
rte_ten = list()
path="/mnt/datadisk0/suyusheng/prompt/prompt/task_prompt_emb/RTEPromptRoberta/"
rte_file = os.listdir(path)
print(rte_file)
for file in rte_file:
    print(torch.load(path+file).shape)
    rte_ten.append(torch.load(path+file))
rte_ten = torch.cat(rte_ten)
print(rte_ten.shape)
rte_ten = rte_ten.reshape(int(rte_ten.shape[0]),int(rte_ten.shape[1]*rte_ten.shape[2]))
print(rte_ten.shape)
#exit()


###########################
###########################
###########################

task_map={0:"sst2",1:"rte"}

sst2_label_ten = torch.zeros(int(sst2_ten.shape[0]),dtype=torch.int32)
rte_label_ten = torch.ones(int(rte_ten.shape[0]),dtype=torch.int32)
print(sst2_label_ten.shape)
print(rte_label_ten.shape)

all_prompt_emb = torch.cat([sst2_ten,rte_ten]).to("cpu").numpy()
all_label = torch.cat([sst2_label_ten,rte_label_ten]).to("cpu").numpy()


#1200 --> 2400 --> 50
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

#sst2_ten = sst2_ten.to("cpu").numpy()

print(all_prompt_emb.shape)

embedding_train = tsne.fit(all_prompt_emb)
#utils_.plot(x=embedding_train, y=all_label, colors=utils_.MOUSE_10X_COLORS, label_map=task_map)
utils_.plot(x=embedding_train, y=all_label, colors=utils_.MOUSE_10X_COLORS, label_map=task_map)



plt.title("Task Prompt Dist")
plt.savefig('output.pdf')











'''

                with torch.no_grad():
                    rep_domain, rep_task = model(input_ids_org=input_ids, sentence_label=label_ids, attention_mask=attention_mask, func="in_domain_task_rep")
            else:
                print("Wrong!!")




            rep = torch.cat([rep_task,rep_domain],-1).to("cpu")

            sentiment_map={"l_neg":1,"l_ne":3,"l_pos":5, "Negative":0,"Neutral":2,"Postive":4}
            for index, tensor in enumerate(rep):
                #aspect, sentiment, tensor
                #if label_ids[index] == 1: #netural
                #    continue
                if aspect_ids[index] == 0:
                    if label_ids[index] == 0:
                        #data_dict['laptop']['negative'].append(tensor)
                        laptop_sentiment_list.append(torch.tensor(1))
                        all_sentiment_list.append(torch.tensor(1))
                    elif label_ids[index] == 1:
                        #data_dict['laptop']['neutral'].append(tensor)
                        laptop_sentiment_list.append(torch.tensor(3))
                        all_sentiment_list.append(torch.tensor(3))
                    elif label_ids[index] == 2:
                        #data_dict['laptop']['positive'].append(tensor)
                        laptop_sentiment_list.append(torch.tensor(5))
                        all_sentiment_list.append(torch.tensor(5))
                    laptop_aspect_list.append(aspect_ids[index])
                    #laptop_sentiment_list.append(label_ids[index])
                    laptop_tensor_list.append(tensor)
                else:
                    if label_ids[index] == 0:
                        #data_dict['restaurant']['negative'].append(tensor)
                        restaurant_sentiment_list.append(torch.tensor(0))
                        all_sentiment_list.append(torch.tensor(0))
                    elif label_ids[index] == 1:
                        #data_dict['restaurant']['neutral'].append(tensor)
                        restaurant_sentiment_list.append(torch.tensor(2))
                        all_sentiment_list.append(torch.tensor(2))
                    elif label_ids[index] == 2:
                        #data_dict['restaurant']['positive'].append(tensor)
                        restaurant_sentiment_list.append(torch.tensor(4))
                        all_sentiment_list.append(torch.tensor(4))
                    restaurant_aspect_list.append(aspect_ids[index])
                    #restaurant_sentiment_list.append(label_ids[index])
                    restaurant_tensor_list.append(tensor)


                all_aspect_list.append(aspect_ids[index])
                #all_sentiment_list.append(label_ids[index])
                all_tensor_list.append(tensor)

        #########
        laptop_aspect_list = torch.stack(laptop_aspect_list).to("cpu").numpy()
        laptop_sentiment_list = torch.stack(laptop_sentiment_list).to("cpu").numpy()
        laptop_tensor_list = torch.stack(laptop_tensor_list).to("cpu").numpy()


        restaurant_aspect_list = torch.stack(restaurant_aspect_list).to("cpu").numpy()
        restaurant_sentiment_list = torch.stack(restaurant_sentiment_list).to("cpu").numpy()
        restaurant_tensor_list = torch.stack(restaurant_tensor_list).to("cpu").numpy()


        all_aspect_list = torch.stack(all_aspect_list).to("cpu").numpy()
        all_sentiment_list = torch.stack(all_sentiment_list).to("cpu").numpy()
        all_tensor_list = torch.stack(all_tensor_list).to("cpu").numpy()
        #########


        #########
        print(laptop_aspect_list.shape)
        print(laptop_sentiment_list.shape)
        print(laptop_tensor_list.shape)
        print("===")

        print(restaurant_aspect_list.shape)
        print(restaurant_sentiment_list.shape)
        print(restaurant_tensor_list.shape)
        print("===")

        print(all_aspect_list.shape)
        #print(all_sentiment_list)
        print(all_sentiment_list.shape)
        print(all_tensor_list.shape)
        print("===")
        #########

        #with open(args.output_dir+".json", "w") as outfile:
        #    json.dump(data_dict, outfile)
        #####Start to draw########
        #emb = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(all_tensor_list)
        #print(emb.shape)

        #plot(all_tensor_list, all_sentiment_list)
        #cosine
        #perplexity
        #400-->1200
        #64
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
        ###
        #embedding_train = tsne.fit(all_tensor_list)
        #utils_.plot(x=embedding_train, y=all_aspect_list, colors=utils_.MOUSE_10X_COLORS, label_map=aspect_map)
        #utils_.plot(x=embedding_train, y=all_sentiment_list, colors=utils_.MOUSE_10X_COLORS, label_map=sentiment_map)
        ###


        ###
        embedding_train = tsne.fit(restaurant_tensor_list)
        utils_.plot(x=embedding_train, y=restaurant_sentiment_list, colors=utils_.MOUSE_10X_COLORS, label_map=sentiment_map)
        ###


        ###
        #embedding_train = tsne.fit(laptop_tensor_list)
        #utils_.plot(x=embedding_train, y=laptop_sentiment_list, colors=utils_.MOUSE_10X_COLORS, label_map=sentiment_map)
        ###
        #plt.savefig(args.output_dir+'.pdf')
        plt.title("Semi-supervised contrastive learning")
        plt.savefig('output.pdf')





if __name__ == "__main__":
    main()
'''
