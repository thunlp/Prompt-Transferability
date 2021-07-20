from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter

class RTEFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        ##########
        self.model_name = config.get("model","model_name")
        if "Roberta" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif "Bert" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            print("Have no matching in the formatter")
            exit()
        #self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        ##########
        self.label2id = {
            "not_entailment": 0,
            "entailment": 1,
        }

    def truncate(self, sent1, sent2):
        while len(sent1) + len(sent2) > self.max_len - 3:
            if len(sent1) > len(sent2):
                sent1.pop()
            else:
                sent2.pop()
        return sent1, sent2

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        for ins in data:
            sent1 = self.tokenizer.encode(ins["sent1"], add_special_tokens = False)
            sent2 = self.tokenizer.encode(ins["sent2"], add_special_tokens = False)
            sent1, sent2 = self.truncate(sent1, sent2)
            tokens = [self.tokenizer.cls_token_id] + sent1 + [self.tokenizer.sep_token_id] + sent2 + [self.tokenizer.sep_token_id]
            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            if mode != "test":
                label.append(self.label2id[ins["label"]])
            inputx.append(tokens)

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }

        return ret
