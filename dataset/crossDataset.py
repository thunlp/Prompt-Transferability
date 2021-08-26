import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset
import random

class crossDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.dataset_list = self.config.get("data","train_dataset_type").lower().split(",")

        self.min_length = []
        self.all_dataset = []
        show_dataset = list()
        #####
        if "re" in self.dataset_list:
            self.re, self.re_length = pre_data_re(mode)
            self.min_length.append(self.re_length)
            self.all_dataset.append(self.re)
            show_dataset.append("re")
            #print("re")
        if "stsb" in self.dataset_list:
            self.stsb, self.stsb_length = pre_data_stsb(mode)
            self.min_length.append(self.stsb_length)
            self.all_dataset.append(self.stsb)
            show_dataset.append("stsb")
            #print("stsb")
        if "sst2" in self.dataset_list:
            self.sst2, self.sst2_length = pre_data_sst2(mode)
            self.min_length.append(self.sst2_length)
            self.all_dataset.append(self.sst2)
            show_dataset.append("sst2")
            #print("sst2")
        if "restaurant" in self.dataset_list:
            self.restaurant, self.restaurant_length = pre_data_restaurant(mode)
            self.min_length.append(self.restaurant_length)
            self.all_dataset.append(self.restaurant)
            show_dataset.append("restaurant")
            #print("restaurant")
        if "qnli" in self.dataset_list:
            self.qnli, self.qnli_length = pre_data_qnli(mode)
            self.min_length.append(self.qnli_length)
            self.all_dataset.append(self.qnli)
            show_dataset.append("qnli")
            #print("qnli")
        if "qqp" in self.dataset_list:
            self.qqp, self.qqp_length = pre_data_qqp(mode)
            self.min_length.append(self.qqp_length)
            self.all_dataset.append(self.qqp)
            show_dataset.append("qqp")
            #print("qqp")
        if "mrpc" in self.dataset_list:
            self.mrpc, self.mrpc_length = pre_data_mrpc(mode)
            self.min_length.append(self.mrpc_length)
            self.all_dataset.append(self.mrpc)
            show_dataset.append("mrpc")
            #print("mrpc")
        if "wnli" in self.dataset_list:
            self.wnli, self.wnli_length = pre_data_wnli(mode)
            self.min_length.append(self.wnli_length)
            self.all_dataset.append(self.wnli)
            show_dataset.append("wnli")
            #print("wnli")
        if "rte" in self.dataset_list:
            self.rte, self.rte_length = pre_data_rte(mode)
            self.min_length.append(self.rte_length)
            self.all_dataset.append(self.rte)
            show_dataset.append("rte")
            #print("rte")
        if "mnli" in self.dataset_list:
            self.mnli, self.mnli_length = pre_data_mnli(mode)
            self.min_length.append(self.mnli_length)
            self.all_dataset.append(self.mnli)
            show_dataset.append("mnli")
            #print("mnli")
        if "laptop" in self.dataset_list:
            self.laptop, self.laptop_length = pre_data_laptop(mode)
            self.min_length.append(self.laptop_length)
            self.all_dataset.append(self.laptop)
            show_dataset.append("laptop")
            #print("laptop")
        if "imdb" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_imdb(mode)
            self.min_length.append(self.imdb_length)
            self.all_dataset.append(self.imdb)
            show_dataset.append("imdb")
            #print("imdb")
        ##
        if "snli" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_snli(mode)
            self.min_length.append(self.snli_length)
            self.all_dataset.append(self.snli)
            show_dataset.append("snli")
            #print("snli")
        if "anli" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_anli(mode)
            self.min_length.append(self.anli_length)
            self.all_dataset.append(self.anli)
            show_dataset.append("anli")
            #print("anli")
        if "recastfactuality" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_recastfactuality(mode)
            self.min_length.append(self.recastfactuality_length)
            self.all_dataset.append(self.recastfactuality)
            show_dataset.append("recastfactuality")
            #print("recastfactuality")
        if "tweetevalsentiment" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_tweetevalsentiment(mode)
            self.min_length.append(self.tweetevalsentiment_length)
            self.all_dataset.append(self.tweetevalsentiment)
            show_dataset.append("tweetevalsentiment")
            #print("tweetevalsentiment")
        if "movierationales" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_movierationales(mode)
            self.min_length.append(self.movierationales_length)
            self.all_dataset.append(self.movierationales)
            show_dataset.append("movierationales")
            #print("movierationales")
        if "emobankarousal" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_emobankarousal(mode)
            self.min_length.append(self.emobankarousal_length)
            self.all_dataset.append(self.emobankarousal)
            show_dataset.append("emobankarousal")
            #print("emobankarousal")
        if "persuasivenessrelevance" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_persuasivenessrelevance(mode)
            self.min_length.append(self.persuasivenessrelevance_length)
            self.all_dataset.append(self.persuasivenessrelevance)
            show_dataset.append("persuasivenessrelevance")
            #print("persuasivenessrelevance")
        if "persuasivenessspecificity" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_persuasivenessspecificity(mode)
            self.min_length.append(self.persuasivenessspecificity_length)
            self.all_dataset.append(self.persuasivenessspecificity)
            show_dataset.append("persuasivenessspecificity")
            #print("persuasivenessspecificity")
        if "emobankdominance" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_emobankdominance(mode)
            self.min_length.append(self.emobankdominance_length)
            self.all_dataset.append(self.emobankdominance)
            show_dataset.append("emobankdominance")
            #print("emobankdominance")
        if "squinkyimplicature" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_squinkyimplicature(mode)
            self.min_length.append(self.squinkyimplicature_length)
            self.all_dataset.append(self.squinkyimplicature)
            show_dataset.append("squinkyimplicature")
            #print("squinkyimplicature")
        if "squinkyformality" in self.dataset_list:
            self.imdb, self.imdb_length = pre_data_squinkyformality(mode)
            self.min_length.append(self.squinkyformality_length)
            self.all_dataset.append(self.squinkyformality)
            show_dataset.append("squinkyformality")
            #print("squinkyformality")

        ###

        ###
        if "agnews_s1" in self.dataset_list:
            self.agnews_mlm_1, self.agnews_mlm_1_length = pre_data_mlm("agnews_s1",mode)
            self.min_length.append(self.agnews_mlm_1_length)
            self.all_dataset.append(self.agnews_mlm_1)
            show_dataset.append("agnews_s1")
        if "agnews_s2" in self.dataset_list:
            self.agnews_mlm_2, self.agnews_mlm_2_length = pre_data_mlm("agnews_s2",mode)
            self.min_length.append(self.agnews_mlm_2_length)
            self.all_dataset.append(self.agnews_mlm_2)
            show_dataset.append("agnews_s2")
            #print("imdb")
        if "cs_wiki_s1" in self.dataset_list:
            self.cs_wiki_mlm_1, self.cs_wiki_mlm_1_length = pre_data_mlm("cs_wiki_s1",mode)
            self.min_length.append(self.cs_wiki_mlm_1_length)
            self.all_dataset.append(self.cs_wiki_mlm_1)
            show_dataset.append("cs_wiki_s1")
        if "cs_wiki_s2" in self.dataset_list:
            self.cs_wiki_mlm_2, self.cs_wiki_mlm_2_length = pre_data_mlm("cs_wiki_s2",mode)
            self.min_length.append(self.cs_wiki_mlm_2_length)
            self.all_dataset.append(self.cs_wiki_mlm_2)
            show_dataset.append("cs_wiki_s2")
            #print("cs_wiki_mlm")
        if "scierc_s1" in self.dataset_list:
            self.scierc_mlm_1, self.scierc_mlm_1_length = pre_data_mlm("scierc_s1",mode)
            self.min_length.append(self.scierc_mlm_1_length)
            self.all_dataset.append(self.scierc_mlm_1)
            show_dataset.append("scierc_s1")
        if "scierc_s2" in self.dataset_list:
            self.scierc_mlm_2, self.scierc_mlm_2_length = pre_data_mlm("scierc_s2",mode)
            self.min_length.append(self.scierc_mlm_2_length)
            self.all_dataset.append(self.scierc_mlm_2)
            show_dataset.append("scierc_s2")
            #print("scierc_mlm")
        if "sst-2_s1" in self.dataset_list:
            self.sst2_mlm_1, self.sst2_mlm_1_length = pre_data_sst2(mode,"SST-2_s1")
            self.min_length.append(self.sst2_mlm_1_length)
            self.all_dataset.append(self.sst2_mlm_1)
            show_dataset.append("sst-2_s1")
        if "sst-2_s2" in self.dataset_list:
            self.sst2_mlm_2, self.sst2_mlm_2_length = pre_data_sst2(mode,"SST-2_s2")
            self.min_length.append(self.sst2_mlm_2_length)
            self.all_dataset.append(self.sst2_mlm_2)
            show_dataset.append("sst-2_s2")
        if "imdb_s1" in self.dataset_list:
            self.imdb_mlm_1, self.imdb_mlm_1_length = pre_data_imdb(mode,"IMDB_s1")
            self.min_length.append(self.imdb_mlm_1_length)
            self.all_dataset.append(self.imdb_mlm_1)
            show_dataset.append("imdb_s1")
        if "imdb_s2" in self.dataset_list:
            self.imdb_mlm_2, self.imdb_mlm_2_length = pre_data_imdb(mode,"IMDB_s1")
            self.min_length.append(self.imdb_mlm_2_length)
            self.all_dataset.append(self.imdb_mlm_2)
            show_dataset.append("imdb_s2")
        ###



        if mode == "train" or mode == "valid":
            self.min_length = min(self.min_length)
        else:
            self.min_length = sum(self.min_length)
        self.all_dataset = self.all_dataset

        show_dataset = list(set(show_dataset))
        show_dataset.sort()
        self.dataset_map_id = {name:id for id, name in enumerate(show_dataset)}
        print("==========")
        print(show_dataset)
        print("==========")
        #exit()

        #self.min_length = min([self.wnli_length, self.re_length, self.stsb_length, self.sst2_length, self.rte_length, self.restaurant_length, self.qqp_length, self.qnli_length, self.mrpc_length, self.mnli_length, self.imdb_length])
        #self.min_length = min([self.wnli_length, self.sst2_length, self.rte_length, self.restaurant_length, self.qqp_length, self.qnli_length, self.mrpc_length, self.mnli_length, self.laptop_length, self.imdb_length])
        '''
        self.min_length = min([self.wnli_length, self.sst2_length, self.restaurant_length, self.qqp_length, self.mrpc_length, self.laptop_length, self.imdb_length])
        '''
        #self.min_length = min([self.laptop_length, self.imdb_length])


        #self.all_dataset = [self.wnli, self.re, self.stsb, self.sst2, self.rte, self.restaurant, self.qqp, self.qnli, self.mrpc, self.mnli, self.laptop, self.imdb]
        #self.all_dataset = [self.wnli, self.sst2, self.rte, self.restaurant, self.qqp, self.qnli, self.mrpc, self.mnli, self.laptop, self.imdb]
        '''
        self.all_dataset = [self.wnli, self.sst2, self.restaurant, self.qqp, self.mrpc, self.laptop, self.imdb]
        '''
        #self.all_dataset = [self.laptop, self.imdb]


        self.all = self.sample_choose()



    def sample_choose(self):
        sample_part = []
        for dataset in self.all_dataset:
            random.shuffle(dataset)
            #self.sample_part += dataset[:self.min_length]
            #print("!!!!")
            #print(dataset[0]["dataset"])
            #print("!!!!")
            sample_part += dataset[:self.min_length]
        return sample_part




    def __getitem__(self, item):
        #return self.all[item]
        return self.all[item], self.dataset_map_id

    def __len__(self):
        #return len(self.data)
        return len(self.all)


#label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:negative, conflict}


def pre_data_mlm(data_name,mode):
    if mode == "train":
        data = json.load(open("data/"+data_name+"/train.json", "r"))
    elif mode == "valid":
        data = json.load(open("data/"+data_name+"/valid.json", "r"))
    else:
        data = json.load(open("data/"+data_name+"/valid.json", "r"))


    key_phrase = list()
    for ins in data:
        #print(ins.keys())
        key_phrase = ins.keys()
        break

    if "tokens" in key_phrase:
        data = [{"text": ins["tokens"].strip(), "label":ins["label"]} for ins in data]
    else:
        pass


    if mode == "test":
        data = [{"sent1": ins["text"].strip(), "dataset": data_name.lower()} for ins in data]
    elif mode == 'valid':
        data = [{"sent1": ins["text"].strip(), "label":ins["label"], "dataset": data_name.lower()} for ins in data]
    else:
        data = [{"sent1": ins["text"].strip(), "label":ins["label"], "dataset": data_name.lower()} for ins in data]

    return data, len(data)






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
    '''
    data = load_dataset('glue','rte')
    '''
    if mode=='valid':
        mode = "validation"
    data = data[mode]
    '''

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
    '''


    dict_={0:1,1:0}
    #dict_={'not_entailment':0,'entailment':1}

    if mode == "test":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip()} for ins in data[1:]]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "dataset":"rte"} for ins in test_data]
    elif mode == "train":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"} for ins in train_data]
    elif mode == "valid":
        #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':int(dict_[ins['label']]), "dataset":"rte"} for ins in validation_data]

    return data, len(data)



def pre_data_qnli(mode):
    data = load_dataset('glue', 'qnli')
    train_data = data['train']
    validation_data = data['validation']
    test_data = data['test']

    dict_={0:1,1:0}

    if mode == "test":
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "dataset":"qnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in train_data]

    '''
    tsv_file = open("data/QNLI/train.tsv",encoding="utf-8-sig")
    train_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QNLI/dev.tsv",encoding="utf-8-sig")
    validation_data = csv.DictReader(tsv_file, delimiter="\t")

    tsv_file = open("data/QNLI/test.tsv",encoding="utf-8-sig")
    test_data = csv.DictReader(tsv_file, delimiter="\t")


    #dict_={0:1,1:0}
    dict_={"not_entailment":0,"entailment":1}

    #data=[]
    if mode == "test":
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "dataset":"qnli"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in validation_data]
    else:
        data = [{"sent1": ins['question'], "sent2": ins['sentence'], "label": dict_[ins['label']] , "dataset":"qnli"} for ins in train_data]
    '''

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
    '''


    if mode == "test":
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'], "dataset":"stsb"} for ins in test_data]
    elif mode == 'valid':
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"stsb"} for ins in validation_data]
    else:
        data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label'], "dataset":"stsb"} for ins in train_data]
    #print(mode, "the number of data", len(data))

    #print([l['label'] for l in data])

    return data, len(data)



def pre_data_sst2(mode,data_name=None):

    if data_name==None:
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


    else:
        _map={0:5, 1:7}
        if mode == "train":
            d = csv.reader(open("./data/"+data_name+"/train.tsv", "r"), delimiter='\t', quotechar='"')
        elif mode == "valid" or mode == "validation":
            d = csv.reader(open("./data/"+data_name+"/dev.tsv", "r"), delimiter='\t', quotechar='"')
        else:
            d = csv.reader(open("./data/"+data_name+"/test.tsv", "r"), delimiter='\t', quotechar='"')

        data = [row for row in d]
        if mode == "test":
            data = [{"sent1": ins[0].strip(), "dataset":data_name.lower()} for ins in data[1:]]
        else:
            data = [{"sent1": ins[0].strip(), "label": _map[int(ins[1].strip())], "dataset":data_name.lower()} for ins in data[1:]]


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

    _dict={"contradiction":0, "neutral":3, "entailment":1}
    '''


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

def pre_data_imdb(mode,data_name=None):

    if data_name == None:
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


    else:
        if mode == "train":
            data_imdb = csv.reader(open("./data/"+data_name+"/train.csv", "r"), delimiter='\t')
        elif mode == "valid":
            data_imdb = csv.reader(open("./data/"+data_name+"/dev.csv", "r"), delimiter='\t')
        else:
            data_imdb = csv.reader(open("./data/"+data_name+"/test.csv", "r"), delimiter='\t')


        data = [row for row in data_imdb]
        #label_map = {"positive":1, "negative":0}
        label_map = {"positive":7, "negative":5}
        if mode == "test":
            data = [{"sent1": ins[0].strip(), "dataset":data_name.lower()} for ins in data]
        else:
            data = [{"sent1": ins[0].strip(), "label":label_map[ins[1].strip()], "dataset":data_name.lower()} for ins in data]

    return data, len(data)


##############
##############

def pre_data_snli(mode):

    if mode == "train":
        data = json.load(open("./data/snli/train.json"))
    elif mode == "valid":
        data = json.load(open("./data/snli/dev.json"))
    else:
        data = json.load(open("./data/snli/test.json"))


    #org_dict = {"contradiction":2,"neutral":1,"entailment":0}
    #after_dict = {"no":0,"neutral":3,"yes":1}
    _dict = {2:0,1:3,0:1}

    if mode == "test":
        self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise']} for ins in data]
    else:
        self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[int(ins['label'])]} for ins in data if int(ins["label"])!=-1]
    print(self.mode, "the number of data", len(self.data))
    # from IPython import embed; embed()


def pre_data_anli(mode):

    if mode == "train":
        data = json.load(open("./data/anli/train_r1.json"))
    elif mode == "valid":
        data = json.load(open("./data/anli/dev_1.json"))
    else:
        data = json.load(open("./data/anli/test_1.json"))


    #org_dict = {"contradiction":2,"neutral":1,"entailment":0}
    #after_dict = {"no":0,"neutral":3,"yes":1}
    _dict = {2:0,1:3,0:1}

    if mode == "test":
        self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise']} for ins in data]
    else:
        self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[int(ins['label'])]} for ins in data if int(ins["label"])!=-1]
    print(self.mode, "the number of data", len(self.data))
    # from IPython import embed; embed()

