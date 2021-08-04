import csv
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from collections import Counter
import heapq

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print("Done")

data_tokens = dict()

datasets = ["SST-2/train.tsv","IMDB/train.csv","cs_wiki/train.json","scierc/train.json","agnews/train.json"]
#datasets = ["IMDB/train.csv"]
#datasets = ["SST-2/train.tsv"]
#datasets = ["cs_wiki/train.json"]
#s1
#s2

#cs: "tokens"
#scierc: "text"
#agnews: "text"

for dataset in datasets:
    data_name, postfix = dataset.strip().split("/")
    for s in ["_s1","_s2"]:
        print("Pre-process:",data_name+s)
        data_dir = "data/"+data_name+s+"/"+postfix
        data_file_check = "data/word_count/"+data_name+s+"/"+postfix+"_check"
        if os.path.isfile(data_file_check):
            print("Already processed. Load from the dir ...")
            with open(data_file_check, encoding="utf-8-sig") as file:
                data_tokens[data_name+s] = json.load(file)


        else:
            with open(data_dir, encoding="utf-8-sig") as file:
                ##.csv or .tsv
                if ".tsv" in data_dir or ".csv" in data_dir:
                    for i, line in enumerate(tqdm(file)):
                        line = line.strip().split("\t")
                        if i == 0:
                            continue
                        else:
                            try:
                                data_tokens[data_name+s] += tokenizer.encode(line[0],add_special_tokens=False)
                            except:
                                data_tokens[data_name+s] = tokenizer.encode(line[0],add_special_tokens=False)

                    data_tokens[data_name+s] = dict(Counter(data_tokens[data_name+s]))

                    file_dir = "data/word_count/"+data_name+s
                    if os.path.isfile(file_dir)==False:
                        os.mkdir(file_dir)
                    else:
                        pass

                    with open(data_file_check, "w") as fp:
                        json.dump(data_tokens[data_name+s], fp)


                elif ".json" in data_dir:
                    file = json.load(file)
                    for i, line in enumerate(tqdm(file)):
                        if "text" in line.keys():
                            try:
                                data_tokens[data_name+s] += tokenizer.encode(line["text"],add_special_tokens=False)
                            except:
                                data_tokens[data_name+s] = tokenizer.encode(line["text"],add_special_tokens=False)


                        elif "tokens" in line.keys():
                            try:
                                data_tokens[data_name+s] += tokenizer.encode(line["tokens"],add_special_tokens=False)

                            except:
                                data_tokens[data_name+s] = tokenizer.encode(line["tokens"],add_special_tokens=False)
                        else:
                            print("Have no key")
                            exit()

                    data_tokens[data_name+s] = dict(Counter(data_tokens[data_name+s]))
                    file_dir = "data/word_count/"+data_name+s
                    if os.path.isfile(file_dir)==False:
                        os.mkdir(file_dir)
                    else:
                        pass

                    with open(data_file_check, "w") as fp:
                        json.dump(data_tokens[data_name+s], fp)
                else:
                    print("Have no file type")
                    exit()

        #########
        print("Done:",data_name+s)
        print("======================")



#################
#################
#choose top N tokens
number_overlap_tokens=5000
token_top = dict()
for data_name, dict_token in data_tokens.items():
    '''
    counter=0
    for w, times in dict_token.items():
        if times!=1:
            counter+=1
    print(data_name, len(dict_token), counter)
    '''
    token_top[data_name] = heapq.nlargest(number_overlap_tokens, dict_token, key=dict_token.get)


#similiarty matrix
data_similarity = dict()
for data_name_A, top_tokens_A in token_top.items():
    for data_name_B, top_tokens_B in token_top.items():
        sim = (len(set(top_tokens_A)&set(top_tokens_B))/number_overlap_tokens)*100
        print("{}, {}, {:.1f}%".format(data_name_A, data_name_B, sim))
        print("------------")
        #print(data_name_A, data_name_B,'%.1f'%()

