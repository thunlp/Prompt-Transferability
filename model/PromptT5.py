import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoConfig
from .modeling_t5 import T5ForConditionalGeneration

class PromptT5(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PromptT5, self).__init__()

        try:
            if config.get("model","model_size")=="large":
                model = "t5-large"
                ckp = "T5LargeForMaskedLM"
                self.hidden_size = 1024
            else:
                model = "t5-base"
                ckp = "T5ForMaskedLM"
                self.hidden_size = 768
        except:
            model = "t5-base"
            ckp = "T5ForMaskedLM"
            self.hidden_size = 768


        #self.init_model_path = config.get('model', 'pretrained_model_path')
        #self.plmconfig = AutoConfig.from_pretrained(self.init_model_path)
        self.plmconfig = AutoConfig.from_pretrained(model)
        # self.plmconfig["architectures"] = ["RobertaForMaskedLM"]
        self.plmconfig.prompt_num = config.getint("prompt", "prompt_num")
        self.plmconfig.prompt_len = config.getint("prompt", "prompt_len")


        if config.get("model","model_size")=="large":
            self.init_model_path = str(ckp)+"/"+"PromptT5Large_init_params"
        else:
            self.init_model_path = str(ckp)+"/"+"PromptT5_init_params"
        ##############
        ###Save a PLM + add prompt -->save --> load again
        #Build model and save it
        if os.path.exists(self.init_model_path+"/pytorch_model.bin"):
            #self.encoder = RobertaForMaskedLM.from_pretrained(self.init_model_path, config=self.plmconfig)
            self.encoder = T5ForConditionalGeneration.from_pretrained(self.init_model_path, config=self.plmconfig)
        else:
            self.encoder = T5ForConditionalGeneration.from_pretrained(model, config=self.plmconfig)

            os.mkdir(self.init_model_path)
            torch.save(self.encoder.state_dict(), str(self.init_model_path)+"/pytorch_model.bin")
            print("Save Done")

            self.encoder = T5ForConditionalGeneration.from_pretrained(self.init_model_path, config=self.plmconfig)



    def init_prompt_emb(self, init_ids, **kwargs):
        self.encoder.roberta.embeddings.init_prompt_emb(torch.tensor(init_ids, dtype=torch.long).to(kwargs['gpu_list'][kwargs['local_rank']]))

    def forward(self, data, config, gpu_list, acc_result, mode, prompt_emb_output=False, **kwargs):
        #print("=====")
        #print(kwargs)
        #print("=====")
        #exit()

        if mode == 'train':
            inputx = []
            mask = []
            labels = []



            if prompt_emb_output == True:
                #Wrong Code
                #output, prompt_emb = self.encoder(input_ids=batch_inputx, attention_mask=batch_mask,prompt_emb_output=prompt_emb_output,prompt_token_len=self.plmconfig.prompt_len)
                print("PromptT5.py line: 102 exit()")
                exit()
                ####
            else:
                #output = self.encoder(input_ids=data["inputx"], labels=data["target"])
                #output = self.encoder(input_ids=data["inputx"], labels=data["target"], attention_mask=data["mask"])
                output = self.encoder(input_ids=data["inputx"], labels=data["target"])
                performance = kwargs["performance"]

                if int(kwargs["step"]%100) == 0:
                    gen = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=config.getint("eval","max_length"))
                    performance = train_acc(gen['sequences'], data["target"])

            #acc_result = acc(output.logits, data["target"], acc_result)

            if prompt_emb_output == True:
                return {'loss': batch_loss}, prompt_emb
            else:
                #return {'loss': batch_loss}
                return {'loss': output["loss"], 'performance':performance}
                #return {'loss': output["loss"]}


        elif mode == 'valid':
            # generated_tokens = self.encoder.generate(input_ids=kwargs['input_ids'],
            #                                          attention_mask=kwargs['attention_mask'],
            #                                          )
            # generated_tokens = self.encoder.generate(input_ids=kwargs['input_ids'],attention_mask=kwargs['attention_mask'], decoder_start_token_id=32099)
            # generated_tokens_no_constrained = self.encoder.generate(input_ids=kwargs['input_ids'],
            #                                          attention_mask=kwargs['attention_mask'],
            #                                          num_beams=kwargs['num_beams'],
            #                                          max_length=kwargs['max_length'],
            #                                          )

            #prefix_allowed_tokens_fn=kwargs['prefix_allowed_tokens_fn']

            #output = self.encoder.generate(input_ids=data["inputx"], )

            output = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=config.getint("eval","max_length"))

            #print(output)
            #exit()


            #generated_tokens = self.encoder.generate(input_ids=kwargs['input_ids'],attention_mask=kwargs['attention_mask'],num_beams=kwargs['num_beams'],output_scores=True,return_dict_in_generate=True)
            #generated_tokens = output['sequences']
            #print(generated_tokens)
            #print("-----")
            #print(generated_tokens.shape)
            #exit()

            acc_result = acc(output['sequences'], data["target"], acc_result)
            return {'acc_result':acc_result}

            # A B C D respectively
            #logits = generated_tokens['scores'][0]
            # Multichoice_logits = logits[:, [71, 272, 205, 309]]
            # correct = []
            # predictions = torch.argmax(Multichoice_logits, dim=-1)
            # for index, i in enumerate(predictions):
            #     correct_per = []
            #     for per_gold in kwargs['gold'][index]:
            #         if i == per_gold:
            #             correct_per.append(1)
            #         else:
            #             correct_per.append(0)
            #     if len(correct_per) != 0:
            #         correct.append(max(correct_per))
            # return correct
            # return generated_tokens




def train_acc(score, label):
    #print("===========")
    #print(score)
    #print("-----------")
    #print(label)
    #print("===========")
    ###
    #score = score[:,2:3]
    score = score[:,1:2]
    ###
    #label = label[:,1:2]
    label = label[:,0:1]
    total = int(label.shape[0])
    #print("===========")
    #print(score)
    #print("----")
    #print(label)
    right = int((score == label).int().sum())
    #print("----")
    #print(int((score == label).int().sum()))
    #print("===========")

    acc_result = round(float(right/total),4)

    return acc_result



def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    #print("===========")
    #print(score)
    #print("-----------")
    #print(label)
    #print("===========")
    ###
    #score = score[:,2:3]
    score = score[:,1:2]
    ###
    #label = label[:,1:2]
    label = label[:,0:1]
    acc_result['total'] += int(label.shape[0])
    #print("===========")
    #print(score)
    #print("----")
    #print(label)
    acc_result['right'] += int((score == label).int().sum())
    #print("----")
    #print(int((score == label).int().sum()))
    #print("===========")

    return acc_result



'''
def acc(scores, the_bests, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    max_indices = []
    for score in scores:
        max_index = torch.argmax(score, dim=-1)
        max_indices.append(max_index.item())
    acc_result['total'] += int(len(the_bests))
    acc_result['right'] += int((np.array(max_indices) == np.array(the_bests)).sum())
    return acc_result
'''
