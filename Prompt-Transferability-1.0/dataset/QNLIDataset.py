import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class QNLIDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset('glue', 'qnli')
        self.train_data = self.data['train']
        self.validation_data = self.data['validation']
        self.test_data = self.data['test']

        #ORG: 1: False, 0:True

        _map={0:1,1:0}
        #Now: 0: False, 1:True

        if mode == "test":
            self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence']} for ins in self.test_data]
        elif mode == 'valid':
            self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]} for ins in self.validation_data]
        else:
            self.data = [{"sent1": ins['question'].strip(), "sent2": ins['sentence'].strip(), "label": _map[ins['label']]} for ins in
                         self.train_data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


