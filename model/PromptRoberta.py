import torch
import torch.nn as nn
import torch.nn.functional as F
import json


import os

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from .modelling_roberta import RobertaForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

class PromptRoberta(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PromptRoberta, self).__init__()
        self.plmconfig = AutoConfig.from_pretrained("roberta-base")
        # self.plmconfig["architectures"] = ["RobertaForMaskedLM"]
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")
        self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        #self.encoder = RobertaForMaskedLM.from_pretrained('roberta-base', self.plmconfig)
        #self.encoder = RobertaForMaskedLM.from_pretrained(self.plmconfig)
        ##############
        ###Save a PLM + add prompt -->save --> load again
        #Build model and save it
        if os.path.exists(self.init_model_path):
            pass
        else:
            from distutils.dir_util import copy_tree
            copy_tree("RobertaForMaskedLM/SST2PromptRoberta", self.init_model_path)
            os.remove(self.init_model_path+"/pytorch_model.bin")
        '''
        self.encoder = RobertaForMaskedLM.from_pretrained('roberta-base')
        torch.save(self.encoder.state_dict(), '../RobertaForMaskedLM/pytorch_model.bin')
        print("===Done===")
        exit()
        '''
        ##############
        #self.encoder = RobertaForMaskedLM.from_pretrained('RobertaForMaskedLM/'+self.init_model, config=self.plmconfig)
        self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)



        # self.encoder = AutoModelForMaskedLM.from_pretrained("roberta-base")
        self.hidden_size = 768
        # self.fc = nn.Linear(self.hidden_size, 2)

        self.criterion = nn.CrossEntropyLoss()
        # self.prompt_num = config.getint("prompt", "prompt_len") # + 1
        # self.init_prompt_emb()

    def init_prompt_emb(self, init_ids):
        self.encoder.roberta.embeddings.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long).to(torch.cuda.current_device()))

        # init_ids = [] #tokenizer.encode("the relation between the first sentence and the second sentence is")
        # pad_num = self.prompt_num - len(init_ids)
        # init_ids.extend([tokenizer.mask_token_id] * pad_num)
        # self.prompt_emb = nn.Embedding(self.prompt_num, self.hidden_size).from_pretrained(self.encoder.get_input_embeddings()(torch.tensor(init_ids, dtype=torch.long)), freeze=False)
        # self.class_token_id = torch.tensor([10932, 2362])

    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output=False, **kwargs):
        # print(self.encoder.roberta.embeddings.prompt_embeddings.weight)
        if prompt_emb_output == True:
            output, prompt_emb = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len)
        else:
            output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'])

        # batch, seq_len = data["inputx"].shape[0], data["inputx"].shape[1]
        # prompt = self.prompt_emb.weight # prompt_len, 768

        # input = self.encoder.get_input_embeddings()(data["inputx"])
        # embs = torch.cat([prompt.unsqueeze(0).repeat(batch, 1, 1), input], dim = 1)

        # output = self.encoder(attention_mask=data['mask'], inputs_embeds=embs)
        logits = output["logits"] # batch, seq_len, vocab_size
        mask_logits = logits[:, 0] # batch, vocab_size
        score = torch.cat([mask_logits[:, 10932].unsqueeze(1), mask_logits[:, 2362].unsqueeze(1)], dim = 1)

        loss = self.criterion(score, data["label"])
        acc_result = acc(score, data['label'], acc_result)

        if prompt_emb_output == True:
            return {'loss': loss, 'acc_result': acc_result}, prompt_emb, data['label']
        else:
            return {'loss': loss, 'acc_result': acc_result}

def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())
    return acc_result
