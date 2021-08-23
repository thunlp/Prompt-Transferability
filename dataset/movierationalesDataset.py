import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class movierationalesDataset(Dataset):
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

        '''
        for line in data:
            print(line)
        print("===")
        print(len(data))
        exit()
        '''

        #emo_dict={"positive":2,"neutral":1,"negative":0,"conflict":3}

        #original: {"positive":1,"negative":0}
        emo_dict={"0":0,"1":1}

        if mode == "test":
            self.data = [{"sent": ins['review'].strip()} for ins in data]
        elif mode == 'valid':
            self.data = [{"sent": ins['review'].strip(), "label": emo_dict[ins['label']]} for ins in data]
        else:
            self.data = [{"sent": ins['review'].strip(), "label": emo_dict[ins['label']]} for ins in data]
        print(self.mode, "the number of data", len(self.data))



    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
