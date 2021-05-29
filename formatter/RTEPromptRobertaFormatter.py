from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter

class RTEPromptRobertaFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.prompt_len = config.getint("prompt", "prompt_len")
        self.prompt_num = config.getint("prompt", "prompt_num")
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.label2id = {
            "not_entailment": 0,
            "entailment": 1,
        }
        self.prompt_prefix = [- (i + 1) for i in range(self.prompt_len)]
        self.prompt_middle = [- (i + 1 + self.prompt_len) for i in range(self.prompt_len)]

    def truncate(self, sent1, sent2):
        while len(sent1) + len(sent2) > self.max_len:
            if len(sent1) > len(sent2):
                sent1.pop()
            else:
                sent2.pop()
        return sent1, sent2

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        max_len = self.max_len + len(self.prompt_middle) + len(self.prompt_prefix) + 3 #+ self.prompt_len * 1 + 4
        # print(len(prefix) + len(middle) + len(end))
        for ins in data:
            sent1 = self.tokenizer.encode(ins["sent1"], add_special_tokens = False)
            sent2 = self.tokenizer.encode(ins["sent2"], add_special_tokens = False)

            sent1, sent2 = self.truncate(sent1, sent2)
            tokens = [self.tokenizer.cls_token_id] + self.prompt_prefix + \
                [self.tokenizer.sep_token_id] + sent1 + self.prompt_middle + \
                sent2 + [self.tokenizer.sep_token_id]

            mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            if mode != "test":
                label.append(self.label2id[ins["label"]])
            inputx.append(tokens)

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return ret
