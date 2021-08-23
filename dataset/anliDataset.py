import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class anliDataset(Dataset):
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
        data = json.load(open(self.data_path), "r")


        #org_dict = {"contradiction":2,"neutral":1,"entailment":0}
        #after_dict = {"contradiction":0,"neutral":1,"entailment":2}
        _dict = {2:0,1:1,0:2}

        if mode == "test":
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise']} for ins in data]
        else:
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": _dict[ins['label']]} for ins in data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


