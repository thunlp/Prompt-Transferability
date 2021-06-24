import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class projectorDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        #self.wnli = pre_data_wnli(mode)
        #self.re = pre_data_re(mode)
        #self.stsb = pre_data_stsb(mode)
        #self.sst2 = pre_data_sst2(mode)
        #self.rte = pre_data_rte(mode)
        self.restaurant = pre_data_restaurant(mode)
        #self.qqp = pre_data_qqp(mode)
        #self.qnli = pre_data_qnli(mode)
        #self.mrpc = pre_data_mrpc(mode)
        #self.mnli = pre_data_mnli(mode)
        self.laptop = pre_data_laptop(mode)
        #self.imdb = pre_data_imdb(mode)


        #self.all = [self.wnli, self.re, self.stsb, self.sst2, self.rte, self.restaurant, self.qqp, self.qnli, self.mrpc, self.mnli, self.laptop, self.imdb]
        self.all = [self.restaurant, self.laptop]


    def __getitem__(self, item):
        #return self.data[item]
        return self.all

    def __len__(self):
        #return len(self.data)
        return len(self.all)





def pre_data_wnli(mode):
    data = load_dataset('glue', 'wnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2']} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in train_data]
    print(mode, "the number of data", len(data))
    return data


def pre_data_re(mode):
    if mode == "train":
        data = json.load(open("./data/RE/train_wiki.json", "r"))
    else:
        data = json.load(open("./data/RE/val_wiki.json", "r"))
    #data = json.load(open(data_path, "r"))
    data = []
    for rel in data:
        if mode == "train":
            inses = data[rel][:int(len(data[rel]) * 0.8)]
        else:
            inses = data[rel][int(len(data[rel]) * 0.8):]
        for ins in inses:
            ins["label"] = rel
            data.append(ins)
    return data


def pre_data_stsb(mode):
    data = load_dataset('glue', 'stsb')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2']} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in train_data]
    print(mode, "the number of data", len(data))
    return data


def pre_data_sst2(mode):
    if mode == "train":
        data = csv.reader(open("./data/SST-2/train.tsv", "r"), delimiter='\t')
    else:
        data = csv.reader(open("./data/SST-2/test.tsv", "r"), delimiter='\t')
    data = [row for row in data]
    if mode == "test":
        data = [{"sent": ins[0].strip()} for ins in data[1:]]
    else:
        data = [{"sent": ins[0].strip(), "label": int(ins[1].strip())} for ins in data[1:]]
    print(mode, "the number of data", len(data))
    return data




def pre_data_rte(mode):
    if mode == "train":
        data = load_dataset('glue', 'rte')
    else:
        data = csv.reader(open("./data/RTE/test.tsv", "r"), delimiter='\t')
    data = [row for row in data]
    if mode == "test":
        data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip()} for ins in data[1:]]
    else:
        data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
    print(mode, "the number of data", len(data))
    return data



def pre_data_restaurant(mode):
    if mode == "train":
        data = json.load(open("./data/restaurant/train.json", "r"))
    else:
        data = json.load(open("./data/restaurant/test.json", "r"))
    emo_dict={"positive":0,"neutral":1,"negative":2,"conflict":3}
    #emo_dict={"positive":0,"neutral":1,"negative":2}

    if mode == "test":
        data = [{"sent": ins['sentence'].strip()} for ins in data]
    elif mode == 'valid':
        data = [{"sent": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']]} for ins in data]
    else:
        data = [{"sent": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']]} for ins in data]
    print(mode, "the number of data", len(data))
    return data


def pre_data_qqp(mode):
    data = load_dataset('glue', 'qqp')
    #data = load_dataset('../data/')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    if mode == "test":
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1']} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": ins['label']} for ins in validation_data]
    else:
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": ins['label']} for ins in
                     train_data]
    print(mode, "the number of data", len(data))
    return data


def pre_data_qnli(mode):
    data = load_dataset('glue', 'qnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    if mode == "test":
        data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence']} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": ins['label']} for ins in validation_data]
    else:
        data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": ins['label']} for ins in train_data]
    print(mode, "the number of data", len(data))
    return data



def pre_data_mrpc(mode):
    data = load_dataset('glue', 'mrpc')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2']} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in train_data]
    print(mode, "the number of data", len(data))
    return data


def pre_data_mnli(mode):
    data = load_dataset('glue', 'mnli')
    train_data = data['train']
    validation_matched_data = data['validation_matched']
    validation_mismatched_data = data['validation_mismatched']
    test_matched_data = data['test_matched']
    test_mismatched_data = data['test_mismatched']

    if mode == "test_matched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise']} for ins in test_matched_data]
    elif mode == "test_mismatched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise']} for ins in test_mismatched_data]
    elif mode == "valid_matched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": ins['label']} for ins in validation_matched_data]
    elif mode == "valid_mismatched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": ins['label']} for ins in validation_mismatched_data]
    else:
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": ins['label']} for ins in train_data]
    print(mode, "the number of data", len(data))
    return data



def pre_data_laptop(mode):
    if mode == "train":
        data = json.load(open("./data/laptop/train.json", "r"))
    else:
        data = json.load(open("./data/laptop/test.json", "r"))
    emo_dict={"positive":0,"neutral":1,"negative":2,"conflict":3}

    if mode == "test":
        data = [{"sent": ins['sentence'].strip()} for ins in data]
    elif mode == 'valid':
        data = [{"sent": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']]} for ins in data]
    else:
        data = [{"sent": ins['sentence'].strip(), "label": emo_dict[ins['sentiment']]} for ins in data]
    print(mode, "the number of data", len(data))
    return data



def pre_data_imdb(mode):
    if mode == "train":
        data_imdb = csv.reader(open("./data/IMDB/train.csv", "r"), delimiter='\t')

    else:
        data_imdb = csv.reader(open("./data/IMDB/test.csv", "r"), delimiter='\t')

    data = [row for row in data_imdb]
    label_map = {"positive":0, "negative":1}
    if mode == "test":
        data = [{"sent": ins[0].strip()} for ins in data]
    else:
        data = [{"sent": ins[0].strip(), "label":label_map[ins[1].strip()]} for ins in data]
    return data
