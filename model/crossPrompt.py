import torch
import torch.nn as nn
import torch.nn.functional as F
import json


import os
import datasets

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from .modelling_roberta import RobertaForMaskedLM

#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
except:
    tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")


#{0: 'imdb', 1: 'laptop', 2: 'mnli', 3: 'mrp', 4: 'qnli', 5: 'qqp', 6: 're', 7: 'restaurant', 8: 'rte', 9: 'sst2', 10: 'stsb', 11: 'wnli'}

def load_task_prompt(model_prompt, config_name, config):
    #choosed_tasks=['imdb','laptop','mnli','mrp','qnli','qqp','re','restaurant','rte','sst2','stsb','wnli']
    #choosed_tasks=['imdb','laptop','mnli','mrp','qnli','qqp','restaurant','rte','sst2','wnli']



    config_name = config_name.split("/")[1].replace(".config","")

    choosed_tasks = config.get("data","train_dataset_type").lower().split(",")
    transfered_model = config.get("model","model_size").lower().split(",")
    model_size = str.title(model_prompt.strip().split("-")[-1])
    model_prompt = str.title(model_prompt.strip().split("-")[0])


    if model_prompt == "Bert":
        model_prompt_not_in = "Roberta"
    elif model_prompt == "Roberta":
        model_prompt_not_in = "Bert"
    print("====")
    print("Include prompt type:",model_prompt)
    print("---")
    print("Not include prompt type:",model_prompt_not_in)
    print("---")
    print("Trained prompt:", model_prompt, model_size)
    print("---")
    print("Transfered model size:", transfered_model)
    print("====")
    #exit()

    name_list = list()
    task_prompt_dict=dict()
    task_prompt_ten=list()
    path="./task_prompt_emb"
    files = os.listdir(path)


    print("----")
    for file in files:

        #cross_mlmPromptRoberta
        if "mlm" in config_name:

            if model_size not in file:
                continue

            if model_prompt not in file:
                continue
            if "mlm" not in file:
                continue
            if "_s1" in file or "_s2" in file:

                #task_prompt_emb = torch.load(path+"/"+file+"/task_prompt")
                task_prompt_emb = torch.load(path+"/"+file+"/task_prompt", map_location=lambda storage, loc:storage)
                name = str(file.strip().split("P")[0]).lower()
                if name=="mr":
                    name+="pc"
                elif name=="qq":
                    name+="p"

                if "_s1" in file:
                    name += "_s1"
                elif "_s2" in file:
                    name += "_s2"

                if name not in choosed_tasks and "sst" not in name:
                    continue

                #if name not in choosed_tasks:
                #    continue
                #print(1)
                print(file)
                name_list.append(name)
                task_prompt_dict[name] = task_prompt_emb
            else:
                continue

        #crossPromptRoberta
        else:
            if model_size == "Base":
                pass
            elif model_size not in file:
                continue
            else:
                print("crossPrompt.py: Line 103 - Have wrong model_size")
                exit()


            if "proj" not in file and model_prompt in file and "mlm" not in file and "_label" not in file and "Large" not in file and "Medium" not in file:
                task_prompt_emb = torch.load(path+"/"+file+"/task_prompt", map_location=lambda storage, loc:storage)
                name = str(file.strip().split("P")[0]).lower()
                if name=="mr":
                    name+="pc"
                elif name=="qq":
                    name+="p"

                if name not in choosed_tasks:
                    continue

                print(file, end='\t')


                name_list.append(name)
                task_prompt_dict[name] = task_prompt_emb
            else:
                continue

    print()
    print("----")
    name_list.sort()
    print(name_list)


    #map_id = {'imdb':0, 'laptop':1, 'mnli':2, 'mrp':3, 'qnli':4, 'qqp':5, 're':6, 'restaurant':7, 'rte':8, 'sst2':9, 'wnli':10}

    #for id, name in name_dict.items():
    for id, name in enumerate(name_list):
        #task_prompt_ten.append(task_prompt_dict[name].to("cuda"))
        task_prompt_ten.append(task_prompt_dict[name])
    task_prompt_ten = torch.stack(task_prompt_ten)
    '''
    for name, id in map_id:
        task_prompt_ten.append(task_prompt_dict[name].to("cuda"))
    task_prompt_ten = torch.stack(task_prompt_ten).to("cuda")
    print(task_prompt_ten.shape)
    exit()
    '''


    return task_prompt_ten



#class crossPromptRoberta(nn.Module):
class crossPrompt(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):

        #super(PromptRoberta, self).__init__()
        #super(projectPromptRoberta, self).__init__()
        #super(crossPromptRoberta, self).__init__()
        super(crossPrompt, self).__init__()


        if "Roberta" in config.get("model","model_base"):
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
        elif "Bert" in config.get("model","model_base"):
            try:
                if config.get("model","model_size")=="large":
                    model = "bert-large"
                    ckp = "BertLargeForMaskedLM"
                    self.hidden_size = 1024
                elif config.get("model","model_size")=="base":
                    model = "bert-base-uncased"
                    ckp = "BertForMaskedLM"
                    self.hidden_size = 768
                elif config.get("model","model_size")=="medium":
                    model = "prajjwal1/bert-medium"
                    ckp = "BertMediumForMaskedLM"
                    self.hidden_size = 512
            except:
                model = "bert-base-uncased"
                ckp = "BertForMaskedLM"
                self.hidden_size = 768
        else:
            print("Wrong!!!")
            print("Wrong!!!")
            print("Wrong!!!")
            print("crossPromptRoberta.py Error")
            exit()

        '''
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
        '''


        self.task_specific_prompt_emb = load_task_prompt(params["model_prompt"],params["args"].config,config).to('cuda')

        self.plmconfig = AutoConfig.from_pretrained(model)
        # self.plmconfig["architectures"] = ["RobertaForMaskedLM"]
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")
        #self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        #self.init_model_path = "RobertaForMaskedLM/"+config.get("data","train_formatter_type")
        #self.init_model_path = str(ckp)+"/"+config.get("data","train_formatter_type")
        '''
        if config.get("model","model_size")=="large":
            self.init_model_path = str(ckp)+"/"+"PromptRobertaLarge_init_params"
        if config.get("model","model_size")=="medium":
            self.init_model_path = str(ckp)+"/"+"PromptRobertaMedium_init_params"
        else:
            self.init_model_path = str(ckp)+"/PromptRoberta_init_params"
        '''


        if "bert-medium" in model:
            model = "bert-medium"

        if config.get("model","model_size")=="large":
            self.init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Large"+"_init_params"
        elif config.get("model","model_size")=="medium":
            self.init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"Medium"+"_init_params"
        else:
            self.init_model_path = str(ckp)+"/"+"Prompt"+str(model.split("-")[0].capitalize())+"_init_params"



        ##############
        ###Save a PLM + add prompt -->save --> load again
        #Build model and save it
        #print(self.init_model_path)
        '''
        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
        else:
            #from distutils.dir_util import copy_tree
            #copy_tree("RobertaForMaskedLM/SST2PromptRoberta", self.init_model_path)
            #copy_tree(str(str(ckp)+"/SST2PromptRoberta"), self.init_model_path)
            #os.remove(self.init_model_path+"/pytorch_model.bin")
            self.encoder = RobertaForMaskedLM.from_pretrained(model, config=self.plmconfig)
            os.mkdir(self.init_model_path)
            torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            #torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            print("Save Done")
            self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
        '''


        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            if "Roberta" in config.get("model","model_base"):
                from .modelling_roberta import RobertaForMaskedLM
                self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            elif "Bert" in config.get("model","model_base"):
                from .modelling_bert import BertForMaskedLM
                self.encoder = BertForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            else:
                print("Wrong")
                exit()
        else:
            if "Roberta" in config.get("model","model_base"):
                from .modelling_roberta import RobertaForMaskedLM
                self.encoder = RobertaForMaskedLM.from_pretrained(model, config=self.plmconfig)
                os.mkdir(self.init_model_path)
                torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
                print("Save Done")
                self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            elif "Bert" in config.get("model","model_base"):
                from .modelling_bert import BertForMaskedLM
                self.encoder = BertForMaskedLM.from_pretrained(model, config=self.plmconfig)
                os.mkdir(self.init_model_path)
                torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
                print("Save Done")
                self.encoder = BertForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            else:
                print("Wrong")
                exit()
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


    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output="replace_task_specific_prompt_emb", **kwargs):



        # print(self.encoder.roberta.embeddings.prompt_embeddings.weight)
        if prompt_emb_output == True:
            output, prompt_emb = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len)


        elif prompt_emb_output == "replace_task_specific_prompt_emb":

            task_specific_prompt_emb = torch.index_select(self.task_specific_prompt_emb, 0, data["task_name"])
            model_AE = kwargs["AE"]

            ###New
            if "100" not in config.get("output","model_name"):
                task_specific_prompt_emb = model_AE(task_specific_prompt_emb)
            else:
                task_specific_prompt_emb_ = task_specific_prompt_emb.reshape( int(task_specific_prompt_emb.shape[0]), int(task_specific_prompt_emb.shape[1])*int(task_specific_prompt_emb.shape[2]))
                task_specific_prompt_emb_ = model_AE(task_specific_prompt_emb_)

                dim_out = int(int(model_AE.decoder.weight.shape[0])/int(task_specific_prompt_emb.shape[1]))
                #dim_out = int(int(model_AE.decoder.weight.shape[0])/100)

                #task_specific_prompt_emb = task_specific_prompt_emb_.reshape(int(task_specific_prompt_emb.shape[0]),int(task_specific_prompt_emb.shape[1]),int(task_specific_prompt_emb.shape[2]))
                task_specific_prompt_emb = task_specific_prompt_emb_.reshape(int(task_specific_prompt_emb.shape[0]),int(task_specific_prompt_emb.shape[1]),dim_out)
                #task_specific_prompt_emb = task_specific_prompt_emb_.reshape(dim_1,dim_2,dim_3)
            ###


            if "mlm" in config.get("output","model_name"):
                output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len, task_specific_prompt_emb=task_specific_prompt_emb, labels=data["label"])
            else:
                output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'], prompt_emb_output=prompt_emb_output, prompt_token_len=self.plmconfig.prompt_len, task_specific_prompt_emb=task_specific_prompt_emb)

        else:
            output = self.encoder(input_ids=data["inputx"], attention_mask=data['mask'])

        # batch, seq_len = data["inputx"].shape[0], data["inputx"].shape[1]
        # prompt = self.prompt_emb.weight # prompt_len, 768

        # input = self.encoder.get_input_embeddings()(data["inputx"])
        # embs = torch.cat([prompt.unsqueeze(0).repeat(batch, 1, 1), input], dim = 1)

        # output = self.encoder(attention_mask=data['mask'], inputs_embeds=embs)


        logits = output["logits"] # batch, seq_len, vocab_size #torch.Size([16, 231, 50265])

        #print("=====")
        #print(config.get("output","model_name"))
        if "mlm" in config.get("output","model_name"):
            loss = output["loss"]
            acc_result = acc_mlm(logits, data['label'], acc_result)

        else:

            mask_logits = logits[:, 0] # batch, vocab_size #torch.Size([16, 50265])

            if config.get("model","model_base") == "Roberta":
                #label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:postive, 8:conflict, 9:low, 10:high}
                score = torch.cat([mask_logits[:,2362].unsqueeze(1), mask_logits[:,10932].unsqueeze(1), mask_logits[:,22303].unsqueeze(1), mask_logits[:,12516].unsqueeze(1),mask_logits[:,29225].unsqueeze(1),mask_logits[:,33407].unsqueeze(1), mask_logits[:, 19397].unsqueeze(1),mask_logits[:,22173].unsqueeze(1),mask_logits[:,17075].unsqueeze(1), mask_logits[:,5481].unsqueeze(1), mask_logits[:,3530].unsqueeze(1)], dim=1)

            elif config.get("model","model_base") == "Bert":
                #label_map={0:no, 1:yes, 2:False, 3:neutral, 4:True, 5:negative, 6:moderate, 7:postive, 8:conflict, 9:low, 10:high}
                score = torch.cat([mask_logits[:,2053].unsqueeze(1), mask_logits[:,2748].unsqueeze(1), mask_logits[:,6270].unsqueeze(1), mask_logits[:,8699].unsqueeze(1),mask_logits[:,2995].unsqueeze(1),mask_logits[:,4997].unsqueeze(1), mask_logits[:,8777].unsqueeze(1),mask_logits[:,3893].unsqueeze(1),mask_logits[:,4736].unsqueeze(1), mask_logits[:,2659].unsqueeze(1), mask_logits[:,2152].unsqueeze(1)], dim=1)

            else:
                print("Cannot access. model/crossPrompt.py Line:373")
                exit()


            loss = self.criterion(score, data["label"])
            acc_result = acc(score, data['label'], acc_result)


        if prompt_emb_output == True:
            return {'loss': loss, 'acc_result': acc_result}, prompt_emb, data['label']
        else:
            return {'loss': loss, 'acc_result': acc_result}



def acc_mlm(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}

    predict = torch.max(score, dim = 2)[1]

    NOT_MASK = [label!=-100]
    predict = predict[NOT_MASK]
    label = label[NOT_MASK]

    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())

    return acc_result


def acc(score, label, acc_result):
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
