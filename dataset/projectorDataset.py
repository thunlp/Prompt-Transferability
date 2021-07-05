import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
import random

class projectorDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        ###
        ##self.re, self.re_length = pre_data_re(mode)
        #self.stsb, self.stsb_length = pre_data_stsb(mode)
        self.sst2, self.sst2_length = pre_data_sst2(mode)
        self.restaurant, self.restaurant_length = pre_data_restaurant(mode)
        self.qnli, self.qnli_length = pre_data_qnli(mode)
        self.qqp, self.qqp_length = pre_data_qqp(mode)
        self.mrpc, self.mrpc_length = pre_data_mrpc(mode)
        self.wnli, self.wnli_length = pre_data_wnli(mode)
        self.rte, self.rte_length = pre_data_rte(mode)
        self.mnli, self.mnli_length = pre_data_mnli(mode)
        self.laptop, self.laptop_length = pre_data_laptop(mode)
        self.imdb, self.imdb_length = pre_data_imdb(mode)
        #print("Done")
        ###


        #self.min_length = min([self.wnli_length, self.re_length, self.stsb_length, self.sst2_length, self.rte_length, self.restaurant_length, self.qqp_length, self.qnli_length, self.mrpc_length, self.mnli_length, self.imdb_length])
        self.min_length = min([self.wnli_length, self.sst2_length, self.rte_length, self.restaurant_length, self.qqp_length, self.qnli_length, self.mrpc_length, self.mnli_length, self.laptop_length, self.imdb_length])
        #self.min_length = min([self.laptop_length, self.imdb_length])


        #self.all_dataset = [self.wnli, self.re, self.stsb, self.sst2, self.rte, self.restaurant, self.qqp, self.qnli, self.mrpc, self.mnli, self.laptop, self.imdb]
        self.all_dataset = [self.wnli, self.sst2, self.rte, self.restaurant, self.qqp, self.qnli, self.mrpc, self.mnli, self.laptop, self.imdb]
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
    data = load_dataset('glue', 'wnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
    #no, yes
    #dict_={0:1,1:0}

    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"wnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"wnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'] , "dataset":"wnli"} for ins in train_data]

    #print([l['label'] for l in data][:10])
    #exit()

    #print(mode, "the number of data", len(data))
    return data, len(data)


def pre_data_rte(mode):
    '''
    if mode == "train":
        d = load_dataset('glue', 'rte')
    else:
        d = csv.reader(open("./data/RTE/test.tsv", "r"), delimiter='\t')
    '''
    data = load_dataset('glue','rte')
    if mode=='valid':
        mode = "validation"
    data = data[mode]

    #data = [row for row in d]
    #print(data)
    #exit()
    #for line in data:
    #    print(line)
    #exit()
    dict_={0:1,1:0}

    if mode == "test":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip()} for ins in data[1:]]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "dataset":"rte"} for ins in data]
    else:
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':dict_[int(ins['label'])], "dataset":"rte"} for ins in data]
    #print(mode, "the number of data", len(data))

    #print(data)
    #print([l['label'] for l in data][:10])
    #exit()

    return data, len(data)


def pre_data_qnli(mode):
    data = load_dataset('glue', 'qnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    dict_={0:1,1:0}
    if mode == "test":
        data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'], "dataset":"qnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": dict_[ins['label']] , "dataset":"qnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": dict_[ins['label']] , "dataset":"qnli"} for ins in train_data]
    #print(mode, "the number of data", len(data))

    #print([l['label'] for l in data][:10])
    #exit()

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
    data = load_dataset('glue', 'stsb')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']
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
    data = load_dataset('glue', 'qqp')
    #data = load_dataset('../data/')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    _map={0:2,1:4}

    if mode == "test":
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'], "dataset":"qqp"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": _map[ins['label']], "dataset":"qqp"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question1'].strip(), "sent2": ins['question1'].strip(), "label": _map[ins['label']], "dataset":"qqp"} for ins in
                     train_data]
    #print(mode, "the number of data", len(data))

    #print([l['label'] for l in data][:10])
    #exit()

    return data, len(data)





def pre_data_mrpc(mode):
    data = load_dataset('glue', 'mrpc')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    _map={0:2,1:4}

    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"mrpc"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"mrpc"} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"mrpc"} for ins in train_data]
    #print(mode, "the number of data", len(data))

    #print([l['label'] for l in data][:10])
    #exit()

    return data, len(data)



def pre_data_mnli(mode):
    data = load_dataset('glue', 'mnli')
    train_data = data['train']
    validation_matched_data = data['validation_matched']
    validation_mismatched_data = data['validation_mismatched']
    test_matched_data = data['test_matched']
    test_mismatched_data = data['test_mismatched']

    #no, neutral, yes
    #_dict={2:0,1:1,0:2}
    _dict={2:0,1:3,0:1}

    if mode == "test_matched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"mnli"} for ins in test_matched_data]
    elif mode == "test_mismatched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'], "dataset":"mnli"} for ins in test_mismatched_data]
    elif mode == "valid_matched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in validation_matched_data]
    elif mode == "valid_mismatched":
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in validation_mismatched_data]
    else:
        data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']], "dataset":"mnli"} for ins in train_data]

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
