import json
import os
from torch.utils.data import Dataset
import csv

class persuasivenessspecificityDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        fin = csv.reader(open(self.data_path, "r"), delimiter="\t", quotechar='"')

        _map = {"low":0, "high":1}

        data = [row for row in fin]
        if mode == "test":
            self.data = [{"sent1": ins[0].strip(), "sent2": ins[1].strip()} for ins in data[1:]]
        else:
            self.data = [{"sent1": ins[0].strip(), "sent2":ins[1].strip(), "label": _map[ins[2].strip()]} for ins in data[1:]]
            '''
            if mode == "valid":
                print(mode)
                print(self.data)
                print("====")
                print(data)
                print(len(data))
                exit()
            '''

        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
