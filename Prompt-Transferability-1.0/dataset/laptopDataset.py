import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class laptopDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        #self.data_path = config.get("data", "train_data_path")
        data = json.load(open(self.data_path, "r"))

        '''
        self.data = []
        for rel in data:
            if mode == "train":
                inses = data[rel][:int(len(data[rel]) * 0.8)]
            else:
                inses = data[rel][int(len(data[rel]) * 0.8):]
            for ins in inses:
                ins["label"] = rel
                self.data.append(ins)
        '''

        emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}

        if mode == "test":
            self.data = [{"sent": ins['sentence'].strip()+ " " + ins["aspect"].strip()} for ins in data]
        elif mode == 'valid':
            self.data = [{"sent": ins['sentence'].strip()+ " " + ins["aspect"].strip(), "label": emo_dict[ins['sentiment']]} for ins in data]
        else:
            self.data = [{"sent": ins['sentence'].strip()+ " " + ins["aspect"].strip(), "label": emo_dict[ins['sentiment']]} for ins in data]
        print(self.mode, "the number of data", len(self.data))



    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
