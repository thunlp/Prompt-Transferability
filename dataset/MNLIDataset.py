import json
import os
from torch.utils.data import Dataset
import csv
from datasets import load_dataset

class MNLIDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data = load_dataset('glue', 'mnli')
        self.train_data = self.data['train']
        self.validation_matched_data = self.data['validation_matched']
        self.validation_mismatched_data = self.data['validation_mismatched']
        self.test_matched_data = self.data['test_matched']
        self.test_mismatched_data = self.data['test_mismatched']

        if mode == "test_matched":
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise']} for ins in self.test_matched_data]
        elif mode == "test_mismatched":
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise']} for ins in
                         self.test_mismatched_data]
        elif mode == "valid_matched":
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": ins['label']} for ins in self.validation_matched_data]
        elif mode == "valid_mismatched":
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": ins['label']}
                         for ins in self.validation_mismatched_data]
        else:
            self.data = [{"sent1": ins['hypothesis'].strip(), "sent2": ins['premise'].strip(), "label": ins['label']} for ins in
                         self.train_data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


