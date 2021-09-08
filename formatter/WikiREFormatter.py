from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter

class WikiREFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        ##########
        self.model_name = config.get("model","model_base")
        if "Roberta" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif "Bert" in self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        else:
            print("Have no matching in the formatter")
            exit()
        #self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        ##########
        self.labelinfo = json.load(open(config.get("data", "label_info"), "r"))

    def sent2token(self, ins):
        ents = [(head[0], head[-1] + 1, 'head') for head in ins["h"][2]] + [(tail[0], tail[-1] + 1, 'tail') for tail in ins["t"][2]]
        ents.sort()
        tokens = [self.tokenizer.cls_token_id]
        lastend = 0
        headpos = -1
        tailpos = -1
        for ent in ents:
            if ent[0] < lastend:
                continue
            text = " ".join(ins["tokens"][lastend: ent[0]])
            tokens += self.tokenizer.encode(text, add_special_tokens=False)
            # "madeupword0000": 50261, "madeupword0001": 50262, "madeupword0002": 50263
            if ent[2] == "head":
                headpos = len(tokens)
                tokens.append(50261)
            else:
                tailpos = len(tokens)
                tokens.append(50263)
            tokens += self.tokenizer.encode(" ".join(ins["tokens"][ent[0]: ent[1]]), add_special_tokens=False)
            if ent[2] == "head":
                tokens.append(50262)
            else:
                tokens.append(49998)
            lastend = ent[1]
        tokens += self.tokenizer.encode(" ".join(ins["tokens"][lastend:]), add_special_tokens=False)
        tokens += [self.tokenizer.sep_token_id]
        assert headpos > 0
        assert tailpos > 0
        if headpos >= self.max_len:
            headpos = 0
        if tailpos >= self.max_len:
            tailpos = 0
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        return tokens, headpos, tailpos

    def process(self, data, config, mode, *args, **params):
        inputx = []
        mask = []
        headpos = []
        tailpos = []
        label = []
        for ins in data:
            tokens, hpos, tpos = self.sent2token(ins)

            mask.append([1] * len(tokens) + [0] * (self.max_len - len(tokens)))

            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_len - len(tokens))
            if mode != "test":
                label.append(self.labelinfo["label2id"][ins["label"]])
            inputx.append(tokens)
            headpos.append(hpos)
            tailpos.append(tpos)

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
            "headpos": torch.tensor(headpos, dtype=torch.long),
            "tailpos": torch.tensor(tailpos, dtype=torch.long),
        }

        return ret
