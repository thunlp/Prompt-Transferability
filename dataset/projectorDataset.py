import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

import random
import sys
csv.field_size_limit(sys.maxsize)

class projectorDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.choose_dataset = config.get("data","train_dataset_type").lower().split(",")

        self.all_length = []
        self.all_dataset = []

        print(self.choose_dataset)

        #####
        if "re" in self.choose_dataset:
            self.re, self.re_length = pre_data_re(mode)
            self.all_length.append(self.re_length)
            self.all_dataset.append(self.re)
            print("re",self.re_length) #del
        if "stsb" in self.choose_dataset:
            self.stsb, self.stsb_length = pre_data_stsb(mode)
            self.all_length.append(self.stsb_length)
            self.all_dataset.append(self.stsb)
            print("stsb",self.stsb_length) #del
            #print("======")
        if "sst2" in self.choose_dataset:
            self.sst2, self.sst2_length = pre_data_sst2(mode)
            self.all_length.append(self.sst2_length)
            self.all_dataset.append(self.sst2)
            print("sst2",self.sst2_length)
        if "restaurant" in self.choose_dataset:
            self.restaurant, self.restaurant_length = pre_data_restaurant(mode)
            self.all_length.append(self.restaurant_length)
            self.all_dataset.append(self.restaurant)
            print("restaurant",self.restaurant_length)
        if "qnli" in self.choose_dataset:
            self.qnli, self.qnli_length = pre_data_qnli(mode)
            self.all_length.append(self.qnli_length)
            self.all_dataset.append(self.qnli)
            print("qnli",self.qnli_length) #del
        if "qqp" in self.choose_dataset:
            self.qqp, self.qqp_length = pre_data_qqp(mode)
            self.all_length.append(self.qqp_length)
            self.all_dataset.append(self.qqp)
            print("qqp",self.qqp_length)
        if "mrpc" in self.choose_dataset:
            self.mrpc, self.mrpc_length = pre_data_mrpc(mode)
            self.all_length.append(self.mrpc_length)
            self.all_dataset.append(self.mrpc)
            print("mrpc",self.mrpc_length)
        if "wnli" in self.choose_dataset:
            self.wnli, self.wnli_length = pre_data_wnli(mode)
            self.all_length.append(self.wnli_length)
            self.all_dataset.append(self.wnli)
            print("wnli",self.wnli_length)
        if "rte" in self.choose_dataset:
            self.rte, self.rte_length = pre_data_rte(mode)
            self.all_length.append(self.rte_length)
            self.all_dataset.append(self.rte)
            print("rte",self.rte_length) #
        if "mnli" in self.choose_dataset:
            self.mnli, self.mnli_length = pre_data_mnli(mode)
            self.all_length.append(self.mnli_length)
            self.all_dataset.append(self.mnli)
            print("mnli",self.mnli_length) #
        if "laptop" in self.choose_dataset:
            self.laptop, self.laptop_length = pre_data_laptop(mode)
            self.all_length.append(self.laptop_length)
            self.all_dataset.append(self.laptop)
            print("laptop",self.laptop_length)
        if "imdb" in self.choose_dataset:
            self.imdb, self.imdb_length = pre_data_imdb(mode)
            self.all_length.append(self.imdb_length)
            self.all_dataset.append(self.imdb)
            print("imdb",self.imdb_length)
            #print("======")
            #print("Done")
            #####
        #print(self.all_length)
        #exit()


        #self.min_length = min([self.wnli_length, self.re_length, self.stsb_length, self.sst2_length, self.rte_length, self.restaurant_length, self.qqp_length, self.qnli_length, self.mrpc_length, self.mnli_length, self.imdb_length])
        #self.min_length = min([self.wnli_length, self.sst2_length, self.rte_length, self.restaurant_length, self.qqp_length, self.qnli_length, self.mrpc_length, self.mnli_length, self.laptop_length, self.imdb_length])

        if mode == "train" or mode == "valid":
            #self.min_length = min([self.wnli_length, self.sst2_length, self.restaurant_length, self.qqp_length, self.mrpc_length, self.laptop_length, self.imdb_length])
            self.min_length = min(self.all_length)
        else:
            self.min_length = sum(self.all_length)
        #self.min_length = min([self.laptop_length, self.imdb_length])


        #self.all_dataset = [self.wnli, self.re, self.stsb, self.sst2, self.rte, self.restaurant, self.qqp, self.qnli, self.mrpc, self.mnli, self.laptop, self.imdb]
        #self.all_dataset = [self.wnli, self.sst2, self.rte, self.restaurant, self.qqp, self.qnli, self.mrpc, self.mnli, self.laptop, self.imdb]
        #self.all_dataset = [self.wnli, self.sst2, self.restaurant, self.qqp, self.mrpc, self.laptop, self.imdb]
        #self.all_dataset = all_dataset
        #self.all_dataset = [self.laptop, self.imdb]


        self.all = self.sample_choose()



    def sample_choose(self):
        sample_part = []
        for dataset in self.all_dataset:
            random.shuffle(dataset)
            #self.sample_part += dataset[:self.min_length]
            sample_part += dataset[:self.min_length]
        return sample_part




    def __getitem__(self, item):
        return self.all[item]

    def __len__(self):
        #return len(self.data)
        return len(self.all)


#label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:negative, conflict}


def pre_data_wnli(mode):
    '''
    data = load_dataset('glue', 'wnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    '''

    tsv_file = open("data/WNLI/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/WNLI/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/WNLI/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")

    #no, yes
    #dict_={0:1,1:0}

    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"wnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": int(ins['label']), "dataset":"wnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": int(ins['label']) , "dataset":"wnli"} for ins in train_data]

    #print([l['label'] for l in data][:10])

    #print(mode, "the number of data", len(data))
    return data, len(data)



def pre_data_rte(mode):
    '''
    if mode == "train":
        d = load_dataset('glue', 'rte')
    else:
        d = csv.reader(open("./data/RTE/test.tsv", "r"), delimiter='\t')
    data = load_dataset('glue','rte')
    if mode=='valid':
        mode = "validation"
    data = data[mode]

    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    '''

    tsv_file = open("data/RTE/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/RTE/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/RTE/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")


    #dict_={0:1,1:0}
    dict_={'not_entailment':0,'entailment':1}

    data=list()
    if mode == "test":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip()} for ins in data[1:]]
        #data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "dataset":"rte"} for ins in test_data]
        for ins in test_data:
            try:
                _d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "dataset":"rte"}
                data.append(_d)
            except:
                pass
    elif mode == "train":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        #data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"} for ins in train_data]
        for ins in train_data:
            try:
                _d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"}
                data.append(_d)
            except:
                pass
    elif mode == "valid":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        #data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"} for ins in validation_data]
        for ins in validation_data:
            try:
                _d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"}
                data.append(_d)
            except:
                pass

    return data, len(data)


#data preprocess problem
def pre_data_qnli(mode):
    '''
    data = load_dataset('glue', 'qnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    dict_={0:1,1:0}
    '''
    tsv_file = open("data/QNLI/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QNLI/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QNLI/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")

    dict_={"not_entailment":0,"entailment":1}

    #data=[]
    if mode == "test":
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "dataset":"qnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in train_data]

    print("Done")
    print(mode, "the number of data", len(data))

    return data, len(data)


######################
######################

def pre_data_re(mode):
    if mode == "train":
        data = json.load(open("./data/RE/train_wiki.json", "r"))
    else:
        data = json.load(open("./data/RE/val_wiki.json", "r"))
    labelinfo = json.load(open("./data/RE/linfo.json", "r"))
    #data = json.load(open(data_path, "r"))
    data_ = []
    for rel in data:
        if mode == "train":
            inses = data[rel][:int(len(data[rel]) * 0.8)]
        else:
            inses = data[rel][int(len(data[rel]) * 0.8):]
        for ins in inses:
            ins["label"] = rel
            data_.append(ins)

    #print(set([l["label"] for l in data_]))
    #exit()

    return data_, len(data_)


def pre_data_stsb(mode):
    '''
    data = load_dataset('glue', 'stsb')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    '''

    tsv_file = open("data/STS-B/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/STS-B/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/STS-B/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")


    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"stsb"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"stsb"} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"stsb"} for ins in train_data]
    #print(mode, "the number of data", len(data))

    #print([l['label'] for l in data])

    return data, len(data)



def pre_data_sst2(mode):

    _map={0:5, 1:7}

    if mode == "train":
        d = csv.reader(open("./data/SST-2/train.tsv", "r"), delimiter='\t', quotechar='"')
    elif mode == "valid" or mode == "validation":
        d = csv.reader(open("./data/SST-2/dev.tsv", "r"), delimiter='\t', quotechar='"')
    else:
        d = csv.reader(open("./data/SST-2/test.tsv", "r"), delimiter='\t', quotechar='"')

    data = [row for row in d]
    if mode == "test":
        data = [{"sent1": ins[0].strip(), "dataset":"sst2"} for ins in data[1:]]
    else:
        data = [{"sent1": ins[0].strip(), "label": _map[int(ins[1].strip())], "dataset":"sst2"} for ins in data[1:]]
    #print(mode, "the number of data", len(data))
    return data, len(data)



def pre_data_restaurant(mode):
    if mode == "train":
        data = json.load(open("./data/restaurant/train.json", "r"))
    elif mode == "valid":
        data = json.load(open("./data/restaurant/test.json", "r"))
    else:
        data = json.load(open("./data/restaurant/test.json", "r"))
    #emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}
    emo_dict={"positive":7,"neutral":6,"negative":5,"conflict":8}
    #emo_dict={"positive":0,"neutral":1,"negative":2}

    if mode == "test":
        data = [{"sent1": ins['sentence'].strip(), "dataset":"restaurant"} for ins in data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']], "dataset":"restaurant"} for ins in data]
    else:
        data = [{"sent1": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']], "dataset":"restaurant"} for ins in data]
    #print(mode, "the number of data", len(data))
    return data, len(data)



def pre_data_qqp(mode):
    '''
    data = load_dataset('glue', 'qqp')
    #data = load_dataset('../data/')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    '''

    tsv_file = open("data/QQP/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QQP/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QQP/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")

    _map={0:2,1:4}

    '''
    if mode == "test":
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'], "dataset":"qqp"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": _map[ins['label']], "dataset":"qqp"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": _map[ins['label']], "dataset":"qqp"} for ins in
                     train_data]
    '''

    if mode == "test":
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'], "dataset":"qqp"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": _map[int(ins['is_duplicate'])], "dataset":"qqp"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": _map[int(ins['is_duplicate'])], "dataset":"qqp"} for ins in
                     train_data]

    #print(mode, "the number of data", len(data))

    #print([l['label'] for l in data][:10])
    #exit()

    return data, len(data)





def pre_data_mrpc(mode):
    '''
    data = load_dataset('glue', 'mrpc')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    '''

    tsv_file = open("data/MRPC/msr_paraphrase_train.txt", encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/MRPC/msr_paraphrase_test.txt", encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/MRPC/msr_paraphrase_test.txt", encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")

    _map={0:2,1:4}

    if mode == "test":
        data = [{"sent1": ins['#1 String'], "sent2": ins['#2 String'], "dataset":"mrpc"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['#1 String'], "sent2": ins['#2 String'], "label": _map[int(ins['Quality'])], "dataset":"mrpc"} for ins in validation_data]
    else:
        data = [{"sent1": ins['#1 String'], "sent2": ins['#2 String'], "label": _map[int(ins['Quality'])], "dataset":"mrpc"} for ins in train_data]
    #print(mode, "the number of data", len(data))

    #print([l['label'] for l in data][:10])
    #exit()

    return data, len(data)



def pre_data_mnli(mode):
    '''
    data = load_dataset('glue', 'mnli')
    train_data = data['train']
    validation_matched_data = data['validation_matched']
    validation_mismatched_data = data['validation_mismatched']
    test_matched_data = data['test_matched']
    test_mismatched_data = data['test_mismatched']
    '''
    tsv_file = open("data/MNLI/train.tsv")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/MNLI/dev_matched.tsv")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/MNLI/dev_matched.tsv")
    test_data = csv.DictReader(tsv_file, delimiter="\t")


    #no, neutral, yes
    #_dict={2:0,1:1,0:2}
    #_dict={2:0,1:3,0:1}
    _dict={"contradiction":0,"neutral":3,"entailment":1}

    data = list()
    if mode == "test_matched":
        #data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"mnli"} for ins in test_matched_data]
        for ins in test_matched_data:
            try:
                d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"mnli"}
                data.append(d)
            except:
                pass
    elif mode == "test_mismatched":
        #data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"mnli"} for ins in test_mismatched_data]
        for ins in test_mismatched_data:
            try:
                d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence1'], "dataset":"mnli"}
                data.append(d)
            except:
                pass
    elif mode == "valid_matched":
        #data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in validation_matched_data]
        for ins in validation_matched_data:
            try:
                d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label1": _dict[ins['label']], "dataset":"mnli"}
                data.append(d)
            except:
                pass
    elif mode == "valid_mismatched":
        #data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in validation_mismatched_data]
        for ins in validation_mismatched_data:
            try:
                d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": _dict[ins['label1']], "dataset":"mnli"}
                data.append(d)
            except:
                pass
    else:
        #data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in train_data]
        for ins in train_data:
            try:
                d = {"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": _dict[ins['label1']], "dataset":"mnli"}
                data.append(d)
            except:
                pass

    #org: [1, 0, 0, 0, 1, 0, 1, 0, 2, 2]
    #print([l['label'] for l in data][:10])
    #exit()

    #print(mode, "the number of data", len(data))
    return data, len(data)



def pre_data_laptop(mode):
    if mode == "train":
        data = json.load(open("./data/laptop/train.json", "r"))
    elif mode == "valid":
        data = json.load(open("./data/laptop/test.json", "r"))
    else:
        data = json.load(open("./data/laptop/test.json", "r"))
    #emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}
    emo_dict={"positive":7,"neutral":6,"negative":5,"conflict":8}

    if mode == "test":
        data = [{"sent1": ins['sentence'].strip(), "dataset":"laptop"} for ins in data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']], "dataset":"laptop"} for ins in data]
    else:
        data = [{"sent1": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']], "dataset":"laptop"} for ins in data]
    #print(mode, "the number of data", len(data))


    return data, len(data)


#label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:postive, 8:conflict}

def pre_data_imdb(mode):
    if mode == "train":
        data_imdb = csv.reader(open("./data/IMDB/train.csv", "r"), delimiter='\t')
    elif mode == "valid":
        data_imdb = csv.reader(open("./data/IMDB/dev.csv", "r"), delimiter='\t')
    else:
        data_imdb = csv.reader(open("./data/IMDB/test.csv", "r"), delimiter='\t')

    data = [row for row in data_imdb]
    #label_map = {"positive":1, "negative":0}
    label_map = {"positive":7, "negative":5}
    if mode == "test":
        data = [{"sent1": ins[0].strip(), "dataset":"imdb"} for ins in data]
    else:
        data = [{"sent1": ins[0].strip(), "label":label_map[ins[1].strip()], "dataset":"imdb"} for ins in data]

    #print([l['label'] for l in data])

    return data, len(data)
