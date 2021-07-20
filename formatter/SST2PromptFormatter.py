from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter

class SST2PromptFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.prompt_len = config.getint("prompt", "prompt_len")
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
    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        mask_place = []
        max_len = self.max_len + 2#+ self.prompt_len * 1 + 4
        for ins in data:
            sent = self.tokenizer.encode(ins["sent"], add_special_tokens = False)
            if len(sent) > max_len:
                sent = sent[:max_len]
            tokens = [self.tokenizer.cls_token_id] + sent + [self.tokenizer.sep_token_id]

            mask.append([1] * self.prompt_len + [1] * len(tokens) + [0] * (max_len - len(tokens)))
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            if mode != "test":
                label.append(ins["label"])
            inputx.append(tokens)

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return ret
