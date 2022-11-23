import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), True)
        self.file_list.sort()

        self.data = []
        for filename in self.file_list:
            self.data = self.data + json.load(open(filename, "r", encoding=encoding))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
