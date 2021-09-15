from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter

class IMDBPromptRobertaFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.prompt_len = config.getint("prompt", "prompt_len")
        self.prompt_num = config.getint("prompt", "prompt_num")
        self.mode = mode
        ##########
        self.model_name = config.get("model","model_base")
        if "Roberta" in self.model_name:
            #self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            except:
                self.tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")
        elif "Bert" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            print("Have no matching in the formatter")
            exit()
        #self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        ##########
        self.prompt_prefix = [- (i + 1) for i in range(self.prompt_len)]
        # self.prompt_middle = [- (i + 1 + self.prompt_len) for i in range(self.prompt_len)]

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        label = []
        #128+3+100=231
        max_len = self.max_len + 3 + self.prompt_num#+ self.prompt_len * 1 + 4
        #print(max_len)
        #exit()
        for ins in data:
            sent = self.tokenizer.encode(ins["sent"], add_special_tokens = False)
            if len(sent) > self.max_len:
                sent = sent[:self.max_len]
            #if len(sent) > max_len:
            #    sent = sent[:max_len]
            tokens = self.prompt_prefix + [self.tokenizer.cls_token_id] + sent + [self.tokenizer.sep_token_id]
            #print("-----")
            #print(len(tokens))
            #print("-----")
            #exit()

            mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            #print(len(mask[0]))
            #print("===")
            #print(len(tokens))
            #print("===")
            if len(tokens) != 231:
                print(tokens)
                print(len(tokens))
                exit()
            #exit()

            if mode != "test":
                label.append(ins["label"])
                #print("====")
                #print(label)
                #print("====")
                #exit()
            inputx.append(tokens)

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return ret
