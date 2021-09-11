import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class RTEDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        '''
        fin = csv.reader(open(self.data_path, "r"), delimiter="\t", quotechar='"')
        '''
        self.data = load_dataset('glue', 'rte')
        #data = data[mode]
        if mode == "train":
            self.data = self.data['train']
        elif mode == "valid":
            self.data = self.data['validation']
        else:
            self.data = self.data['test']

        dict_ = {0:1,1:0}
        #data = [row for row in fin]
        if mode == "test":
            #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip()} for ins in data[1:]]
            self.data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip()} for ins in self.data]
        else:
            #data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
            #self.data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':dict_[int(ins['label'])]} for ins in self.data]
            self.data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), 'label':dict_[int(ins['label'])]} for ins in self.data]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
