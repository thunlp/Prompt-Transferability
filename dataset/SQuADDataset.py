import json
import os
from torch.utils.data import Dataset
import csv
import numpy as np
import pandas as pd

class SQuADDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        # fin = csv.reader(open(self.data_path, "r", encoding='utf-8'), delimiter="\t", quotechar='"')
        self.df = pd.read_csv(self.data_path, sep='\t')
        #columns=['id', 'title', 'context', 'question', 'text', 'answer_start']
        # if mode == "test":
        #     self.data = [{"sent": ins[0].strip()} for ins in data[1:]]
        # else:

        self.data = []
        for ins in self.df.iterrows():
            if not isinstance(ins[1]['text'], str):
                continue
                tmp = {"id": ins[1]['id'].strip(), "title": ins[1]['title'].strip(), "context": ins[1]['context'].strip(),
                       "question": ins[1]['question'].strip(),
                       "text": '', "answer_start": ''}
            else:
                tmp = {"id": ins[1]['id'].strip(), "title": ins[1]['title'].strip(), "context": ins[1]['context'].strip(),
                       "question": ins[1]['question'].strip(),
                       "text": ins[1]['text'].strip(), "answer_start": int(ins[1]['answer_start'])}
            self.data.append(tmp)

        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


