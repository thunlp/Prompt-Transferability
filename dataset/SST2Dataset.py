import json
import os
from torch.utils.data import Dataset
import csv

class SST2Dataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        fin = csv.reader(open(self.data_path, "r"), delimiter="\t", quotechar='"')

        data = [row for row in fin]

        #if "T5" in self.config.get("model","model_base"):
        #    if mode == "test":
        #        self.data = [{"sent": ins[0].strip()} for ins in data[1:]]
        #    else:
        #        self.data = [{"sent": ins[0].strip(), "label": int(ins[1].strip())} for ins in data[1:]]
        #else:
        if mode == "test":
            self.data = [{"sent": ins[0].strip()} for ins in data[1:]]
        else:
            self.data = [{"sent": ins[0].strip(), "label": int(ins[1].strip())} for ins in data[1:]]
        print(self.mode, "the number of data", len(self.data))
        # from IPython import embed; embed()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
