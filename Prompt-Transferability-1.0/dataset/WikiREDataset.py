import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class WikiREDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        #self.data_path = config.get("data", "train_data_path")
        data = json.load(open(self.data_path, "r"))
        self.data = []
        for rel in data:
            if mode == "train":
                inses = data[rel][:int(len(data[rel]) * 0.8)]
            else:
                inses = data[rel][int(len(data[rel]) * 0.8):]
            for ins in inses:
                ins["label"] = rel
                self.data.append(ins)


    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
