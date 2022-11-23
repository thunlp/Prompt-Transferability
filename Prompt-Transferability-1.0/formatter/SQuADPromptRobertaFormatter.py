from transformers import AutoTokenizer
import torch
import json
import numpy as np
from .Basic import BasicFormatter

class SQuADPromptRobertaFormatter(BasicFormatter):
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
        token_types_id = []
        start_id = []
        end_id = []
        text = []
        max_len = self.max_len + 3 + self.prompt_num#+ self.prompt_len * 1 + 4
        for ins in data:
            question = self.tokenizer.encode(ins["question"], add_special_tokens = False)
            context_obj = self.tokenizer(ins["context"], add_special_tokens = False, return_offsets_mapping=True)
            context = self.tokenizer.encode(ins["context"], add_special_tokens = False)


            if len(question) + len(context) > max_len:
                context = context[:max_len - len(question)]
            tokens = self.prompt_prefix + [self.tokenizer.cls_token_id] + question + [self.tokenizer.sep_token_id] + context + [self.tokenizer.sep_token_id]
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            mask.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
            prefix_len = self.prompt_len + len(question) + 2
            if mode != "test":
                if len(ins['text']) == 0:
                    start_id.append(self.prompt_len)
                    end_id.append(self.prompt_len)
                else:
                    start_token = 0
                    end_token = len(context) - 1
                    while context_obj['offset_mapping'][start_token][0] < ins['answer_start'] and start_token < len(context) - 1:
                        start_token = start_token + 1
                    start_id.append(start_token + prefix_len)
                    while context_obj['offset_mapping'][end_token][0] > ins['answer_start'] + len(ins['text']):
                        end_token = end_token - 1
                    end_id.append(end_token + prefix_len)
            inputx.append(tokens)
            # try:
            #     assert len([0] * (self.prompt_len + len(question) + 2) + [1] * (max_len - (self.prompt_len + len(question) + 2))) == len(tokens)
            # except:
            #     print(len(tokens), len([0] * (self.prompt_len + len(question) + 2) + [1] * (max_len - (self.prompt_len + len(question) + 2))))
            # token_types_id.append([0] * (self.prompt_len + len(question) + 2) + [1] * (max_len - (self.prompt_len + len(question) + 2)))

            text.append(ins["text"])

        ret = {
            "inputx": torch.tensor(inputx, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.float),
            "start_id": torch.tensor(start_id, dtype=torch.long),
            "end_id": torch.tensor(end_id, dtype=torch.long),
            "text": text
        }
        return ret
