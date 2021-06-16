import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class MRPCDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset('glue', 'mrpc')
        self.train_data = self.data['train']
        self.validation_data = self.data['validation']
        self.test_data = self.data['test']

        if mode == "test":
            self.data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2']} for ins in self.test_data]
        elif mode == 'valid':
            self.data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in self.validation_data]
        else:
            self.data = [{"sent1": ins['sentence1'].strip(), "sent2": ins['sentence2'].strip(), "label": ins['label']} for ins in
                         self.train_data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


