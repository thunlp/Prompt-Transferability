import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class recastpunsDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        '''
        self.data = load_dataset('glue', 'mnli')
        self.train_data = self.data['train']
        self.validation_matched_data = self.data['validation_matched']
        self.validation_mismatched_data = self.data['validation_mismatched']
        self.test_matched_data = self.data['test_matched']
        self.test_mismatched_data = self.data['test_mismatched']
        '''
        self.data_path = config.get("data", "%s_data_path" % mode)
        data = json.load(open(self.data_path))

        #org: [not-entailed, entailed]
        _dict = {"not-entailed":0,"entailed":1}

        if mode == "test":
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['context']} for ins in data]
        else:
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['context'].strip(), "label": _dict[ins['label']]} for ins in data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


