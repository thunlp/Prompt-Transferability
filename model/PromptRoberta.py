import torch
import torch.nn as nn
import torch.nn.functional as F
import json


import os
import datasets

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from .modelling_roberta import RobertaForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

class PromptRoberta(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PromptRoberta, self).__init__()

        #Roberta variant sizes model: https://huggingface.co/readerbench/RoBERT-small
        #Model	Weights	L	H	A	MLM accuracy	NSP accuracy
        #RoBERT-small	19M	12	256	8	0.5363	0.9687
        #RoBERT-base	114M	12	768	12	0.6511	0.9802
        #RoBERT-large	341M	24	1024	24	0.6929	0.9843


        try:
            if config.get("model","model_size")=="large":
                model = "roberta-large"
                ckp = "RobertaLargeForMaskedLM"
                self.hidden_size = 1024
            else:
                model = "roberta-base"
                ckp = "RobertaForMaskedLM"
                self.hidden_size = 768
        except:
            model = "roberta-base"
            ckp = "RobertaForMaskedLM"
            self.hidden_size = 768


        self.plmconfig = AutoConfig.from_pretrained(model)
        # self.plmconfig["architectures"] = ["RobertaForMaskedLM"]
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")
        #self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        #self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        #self.init_model_path = str(ckp)+"/"+config.get("data","train_formatter_type")
        if config.get("model","model_size")=="large":
            self.init_model_path = str(ckp)+"/"+"PromptRobertaLarge_init_params"
        else:
            self.init_model_path = str(ckp)+"/"+"PromptRoberta_init_params"
        ##############
        ###Save a PLM + add prompt -->save --> load again
        #Build model and save it
        #print("====")
        #print(self.init_model_path)
        #print("====")
        #exit()
        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
        else:
            #from distutils.dir_util import copy_tree
            #copy_tree("RobertaForMaskedLM/SST2PromptRoberta", self.init_model_path)
            #copy_tree(str(str(ckp)+"/SST2PromptRoberta"), self.init_model_path)
            #os.remove(self.init_model_path+"/pytorch_model.bin")

            #print("==========")
            #print("==========")
            #print(self.plmconfig)
            #print(type(self.plmconfig))
            #exit()
            #print(111111111)
            self.encoder = RobertaForMaskedLM.from_pretrained(model, config=self.plmconfig)
            #exit()
            #print(self.encoder.state_dict().keys())
            os.mkdir(self.init_model_path)
            torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            print("Save Done")
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            #print(1111111)
            #print(self.encoder.state_dict().keys())
            #exit()

        ##############
        #self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)


        # self.encoder = AutoModelForMaskedLM.from_pretrained("roberta-base")
        #self.hidden_size = 768
        # self.fc = nn.Linear(self.hidden_size, 2)
        if config.get("data", "train_dataset_type") == "STSB":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        # self.prompt_num = config.getint("prompt", "prompt_len") # + 1
        # self.init_prompt_emb()

        #Refer to https://github.com/xcjthu/prompt/blob/master/model/PromptRoberta.py : line31 revised
        #self.labeltoken = torch.tensor([10932, 2362], dtype=torch.long)
        #self.softlabel = config.getboolean("prompt", "softlabel")
        #if self.softlabel:
        #    self.init_softlabel(self.plmconfig.vocab_size, len(self.labeltoken))

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


        logits = output["logits"] # batch, seq_len, vocab_size #torch.Size([16, 231, 50265])

        mask_logits = logits[:, 0] # batch, vocab_size #torch.Size([16, 50265])


        '''
        print("==============")
        print("==============")

        #sentiment
        #mo_dict={"positive":0,"neutral":1,"negative":2,"conflict":3}
        print(tokenizer.encode("positive",add_special_tokens=False)) #22173
        #print(tokenizer.encode("neutral",add_special_tokens=False)) #12516
        print(tokenizer.encode("moderate",add_special_tokens=False)) #19397
        print(tokenizer.encode("negative",add_special_tokens=False)) #33407
        print(tokenizer.encode("conflict",add_special_tokens=False)) #'conf':17075,, 'lict':

        #NLI
        print(tokenizer.convert_ids_to_tokens([10932])) #['yes']
        print(tokenizer.convert_ids_to_tokens([12516])) #['neutral']
        print(tokenizer.convert_ids_to_tokens([2362])) #['no']

        #paraphrase
        print(tokenizer.encode("true",add_special_tokens=False)) #[29225]
        print(tokenizer.encode("false",add_special_tokens=False)) #[22303]


        print(tokenizer.encode("right",add_special_tokens=False)) #[4070]
        print(tokenizer.encode("wrong",add_special_tokens=False)) #[35621]

        #Discourse
        #(["high”]): 3530
        #(["low”]): 5481

        print("==============")
        print("==============")
        exit()
        '''

        if config.get("data", "train_dataset_type") == "laptop" or config.get("data", "train_dataset_type") == "restaurant":
            #sentiment
            #mo_dict={"positive":22173,"moderate":19397,"negative":33407,"conflict":17075}
            score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:, 19397].unsqueeze(1), mask_logits[:, 22173].unsqueeze(1), mask_logits[:,17075].unsqueeze(1)], dim=1)

        elif config.get("data", "train_dataset_type") == "tweetevalsentiment":
            #sentiment
            #mo_dict={"positive":22173,"moderate":19397,"negative":33407}
            score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:, 19397].unsqueeze(1), mask_logits[:, 22173].unsqueeze(1)], dim=1)

        elif config.get("data", "train_dataset_type") == "SST2" or config.get("data", "train_dataset_type") == "IMDB" or config.get("data", "train_dataset_type") == "movierationales":
            #sentiment
            #mo_dict={"positive":22173,"negative":33407}
            score = torch.cat([mask_logits[:, 33407].unsqueeze(1), mask_logits[:,22173].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "MNLI" or config.get("data", "train_dataset_type") == "snli" or config.get("data", "train_dataset_type") == "anli":
            #NLI
            #mo_dict={"yes":10932,"neutral":12516,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 12516].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "RTE" or "recast" in config.get("data", "train_dataset_type"):
            #NLI
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "WNLI":
            #NLI
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "QNLI":
            #NLI
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "MRPC":
            #paraphrase
            #mo_dict={"true":29225,"false":22303}
            score = torch.cat([mask_logits[:, 22303].unsqueeze(1), mask_logits[:,29225].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "QQP":
            #paraphrase
            #mo_dict={"true":29225,"false":22303}
            score = torch.cat([mask_logits[:, 22303].unsqueeze(1), mask_logits[:,29225].unsqueeze(1)], dim=1)
        elif config.get("data", "train_dataset_type") == "STSB":
            score = mask_logits[:, 1032]
        elif config.get("data", "train_dataset_type") == "emobankarousal" or config.get("data", "train_dataset_type") == "persuasivenessrelevance" or config.get("data", "train_dataset_type") == "persuasivenessspecificity" or config.get("data", "train_dataset_type") == "emobankdominance" or config.get("data", "train_dataset_type") == "squinkyimplicature" or config.get("data", "train_dataset_type") == "squinkyformality":
            score = torch.cat([mask_logits[:,5481].unsqueeze(1), mask_logits[:,3530].unsqueeze(1)], dim=1)
        elif "ethics" in config.get("data", "train_dataset_type"):
            #"acceptable":[32047], "un":[879]
            score = torch.cat([mask_logits[:, 897].unsqueeze(1), mask_logits[:,32047].unsqueeze(1)], dim=1)
        else:
            #Other
            print("PromptRoberta: What is this task?")
            #mask_logits:torch.Size([16, 50265])
            #mo_dict={"yes":10932,"no":2362}
            score = torch.cat([mask_logits[:, 2362].unsqueeze(1), mask_logits[:, 10932].unsqueeze(1)], dim=1)




        loss = self.criterion(score, data["label"])
        if config.get("data", "train_dataset_type") == "STSB":
            acc_result = pearson(score, data['label'], acc_result)
        else:
            acc_result = acc(score, data['label'], acc_result)

        if prompt_emb_output == True:
            return {'loss': loss, 'acc_result': acc_result}, prompt_emb, data['label']
        else:
            return {'loss': loss, 'acc_result': acc_result}



def acc(score, label, acc_result):
    '''
    print("========")
    print("========")
    print(label)
    print(score)
    #print(predict)
    print("========")
    print("========")
    exit()
    '''
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())

    return acc_result


def pearson(score, label, acc_result):
    stsb_result = cal_pearson(score, label)
    if acc_result is None:
        acc_result = {'total_pearson': 0, 'batch_num': 0}
    acc_result['total_pearson'] += stsb_result['pearson']
    acc_result['batch_num'] += 1
    return acc_result


def cal_pearson(score, label):
    tmp_result = {}
    score_bar = torch.mean(score, dim=-1)
    label_bar = torch.mean(label, dim=-1)
    numerator = torch.sum(torch.mul(score-score_bar, label - label_bar), dim=-1)
    denominator = torch.sqrt(torch.sum((score-score_bar) ** 2, dim=-1)) * torch.sqrt(torch.sum((label-label_bar) ** 2, dim=-1))
    pearson_result = numerator / denominator
    tmp_result['pearson'] = pearson_result.item()
    return tmp_result
