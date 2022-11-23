import json
import os
from torch.utils.data import Dataset
import csv

class ethicsvirtueDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        fin = csv.reader(open(self.data_path, "r"), delimiter=",")

        data = [row for row in fin if row[0]=='1' or row[0]=='0']
        if mode == "test":
            self.data = [{"sent": ins[1].strip()} for ins in data]
        else:
            self.data = [{"sent": ins[1].strip(), "label": int(ins[0].strip())} for ins in data]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
