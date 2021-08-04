import csv
import json
import random
import os


'''
SST-2/train.tsv 67349
wrong_data 0
=======================
IMDB/train.csv 24985
wrong_data 14
=======================
cs_wiki/train.json 10552
wrong_data 0
=======================
scierc/train.jsonl 3219
wrong_data 0
=======================
agnews/train.jsonl 115000
wrong_data 0
=======================
'''

max_length=20000
datasets = ["SST-2/train.tsv","IMDB/train.csv","cs_wiki/train.json","scierc/train.jsonl","agnews/train.jsonl"]
#datasets = ["scierc/train.jsonl"]
for dataset in datasets:
    title = []
    length = []
    data_len = 0
    wrong_data = 0
    s1=[]
    s2=[]
    ###############
    if "SST-2" in dataset or "IMDB" in dataset:
        with open("data/"+dataset, encoding="utf-8-sig") as file:
            for i, line in enumerate(file):
                line = line.strip().split("\t")
                if i == 0:
                    title = line
                    data_len = len(title)
                    continue
                if len(line) != data_len:
                    wrong_data+=1
                else:
                    _dict=dict()
                    for id, key in enumerate(title):
                        _dict[key] = line[id]
                    #length.append(line)
                    length.append(_dict)

        print(dataset, len(length))
        print("wrong_data", wrong_data)
        epochs = int(max_length/len(length))
        if epochs < 1:
            s1 += random.sample(length,max_length)
            s2 += random.sample(length,max_length)
        else:
            while epochs>=1:
                s1 += length
                s2 += length
                epochs -= 1
            s1 += random.sample(length,max_length-len(s1))
            s2 += random.sample(length,max_length-len(s2))
        print("s1",len(s1))
        print("s2",len(s2))



        #save to tsv, csv
        data_name, postfix = dataset.split("/")
        dir = "data/"+data_name+"_s1/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w', newline='') as f:
            #writer = csv.DictWriter(f, delimiter='\t', title)
            writer = csv.DictWriter(f, title, delimiter="\t")
            writer.writeheader()
            writer.writerows(s1)
        dir = "data/"+data_name+"_s2/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w', newline='') as f:
            #writer = csv.DictWriter(f, delimiter='\t', title)
            writer = csv.DictWriter(f, title, delimiter="\t")
            writer.writeheader()
            writer.writerows(s1)
        print("=======================")



    elif "cs_wiki" in dataset:
        with open("data/"+dataset, encoding="utf-8-sig") as file:
            file = json.load(file)
            for line in file:
                length.append(line)
        print(dataset, len(length))
        print("wrong_data", wrong_data)
        epochs = int(max_length/len(length))
        if epochs < 1:
            s1 += random.sample(length,max_length)
            s2 += random.sample(length,max_length)
        else:
            while epochs>=1:
                s1 += length
                s2 += length
                epochs -= 1
            s1 += random.sample(length,max_length-len(s1))
            s2 += random.sample(length,max_length-len(s2))
        print("s1",len(s1))
        print("s2",len(s2))


        #save to json
        data_name, postfix = dataset.split("/")
        dir = "data/"+data_name+"_s1/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(s1, f)

        dir = "data/"+data_name+"_s2/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(s2, f)
        print("=======================")


    elif "scierc" in dataset or  "agnews" in dataset:
        with open("data/"+dataset, encoding="utf-8-sig") as file:
            for line in file:
                line = json.loads(line)
                length.append(line)
        print(dataset, len(length))
        print("wrong_data", wrong_data)
        epochs = int(max_length/len(length))
        if epochs < 1:
            s1 += random.sample(length,max_length)
            s2 += random.sample(length,max_length)
        else:
            while epochs>=1:
                s1 += length
                s2 += length
                epochs -= 1
            s1 += random.sample(length,max_length-len(s1))
            s2 += random.sample(length,max_length-len(s2))
        print("s1",len(s1))
        print("s2",len(s2))


        #save to json
        data_name, postfix = dataset.split("/")
        postfix = "train.json"
        dir = "data/"+data_name+"_s1/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(s1, f)

        dir = "data/"+data_name+"_s2/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(s2, f)
        print("=======================")


    ###############
















max_length=20000
datasets = ["SST-2/dev.tsv","IMDB/dev.csv","cs_wiki/valid.json","scierc/dev.jsonl","agnews/dev.jsonl"]
for dataset in datasets:
    title = []
    length = []
    data_len = 0
    wrong_data = 0
    s1=[]
    s2=[]
    ###############
    if "SST-2" in dataset or "IMDB" in dataset:
        with open("data/"+dataset, encoding="utf-8-sig") as file:
            for i, line in enumerate(file):
                line = line.strip().split("\t")
                if i == 0:
                    title = line
                    data_len = len(title)
                    continue
                if len(line) != data_len:
                    wrong_data+=1
                else:
                    _dict=dict()
                    for id, key in enumerate(title):
                        _dict[key] = line[id]
                    #length.append(line)
                    length.append(_dict)

        print(dataset, len(length))
        print("wrong_data", wrong_data)
        '''
        epochs = int(max_length/len(length))
        if epochs < 1:
            s1 += random.sample(length,max_length)
            s2 += random.sample(length,max_length)
        else:
            while epochs>=1:
                s1 += length
                s2 += length
                epochs -= 1
            s1 += random.sample(length,max_length-len(s1))
            s2 += random.sample(length,max_length-len(s2))
        print("s1",len(s1))
        print("s2",len(s2))
        '''



        #save to tsv, csv
        data_name, postfix = dataset.split("/")
        dir = "data/"+data_name+"_s1/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w', newline='') as f:
            #writer = csv.DictWriter(f, delimiter='\t', title)
            writer = csv.DictWriter(f, title, delimiter="\t")
            writer.writeheader()
            writer.writerows(length)
        dir = "data/"+data_name+"_s2/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w', newline='') as f:
            #writer = csv.DictWriter(f, delimiter='\t', title)
            writer = csv.DictWriter(f, title, delimiter="\t")
            writer.writeheader()
            writer.writerows(length)
        print("=======================")



    elif "cs_wiki" in dataset:
        with open("data/"+dataset, encoding="utf-8-sig") as file:
            file = json.load(file)
            for line in file:
                length.append(line)
        print(dataset, len(length))
        print("wrong_data", wrong_data)
        '''
        epochs = int(max_length/len(length))
        if epochs < 1:
            s1 += random.sample(length,max_length)
            s2 += random.sample(length,max_length)
        else:
            while epochs>=1:
                s1 += length
                s2 += length
                epochs -= 1
            s1 += random.sample(length,max_length-len(s1))
            s2 += random.sample(length,max_length-len(s2))
        print("s1",len(s1))
        print("s2",len(s2))
        '''


        #save to json
        data_name, postfix = dataset.split("/")
        dir = "data/"+data_name+"_s1/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(length, f)

        dir = "data/"+data_name+"_s2/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(length, f)
        print("=======================")


    elif "scierc" in dataset or  "agnews" in dataset:
        with open("data/"+dataset, encoding="utf-8-sig") as file:
            for line in file:
                line = json.loads(line)
                length.append(line)
        print(dataset, len(length))
        print("wrong_data", wrong_data)
        '''
        epochs = int(max_length/len(length))
        if epochs < 1:
            s1 += random.sample(length,max_length)
            s2 += random.sample(length,max_length)
        else:
            while epochs>=1:
                s1 += length
                s2 += length
                epochs -= 1
            s1 += random.sample(length,max_length-len(s1))
            s2 += random.sample(length,max_length-len(s2))
        print("s1",len(s1))
        print("s2",len(s2))
        '''


        #save to json
        data_name, postfix = dataset.split("/")
        postfix = "valid.json"
        dir = "data/"+data_name+"_s1/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(length, f)

        dir = "data/"+data_name+"_s2/"
        if os.path.isdir(dir):
            pass
        else:
            os.mkdir(dir)
        with open(dir+postfix, 'w') as f:
            json.dump(length, f)
        print("=======================")


    ###############

