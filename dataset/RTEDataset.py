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
        fin = csv.reader(open(self.data_path, "r"), delimiter="\t", quotechar='"')
        '''
        self.data = load_dataset('glue', 'rte')
        self.train_data = self.data['train']
        self.validation_data = self.data['validation']
        self.test_data = self.data['test']
        '''


        data = [row for row in fin]
        if mode == "test":
            self.data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip()} for ins in data[1:]]
        else:
            self.data = [{"sent1": ins[1].strip(), "sent2": ins[2].strip(), "label": ins[3].strip()} for ins in data[1:] if len(ins) == 4]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
