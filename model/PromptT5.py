import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoConfig
from .modeling_t5 import T5ForConditionalGeneration
from torchnlp.metrics import get_moses_multi_bleu

from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained("T5ForMaskedLM/t5-base")

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoother = SmoothingFunction()

class PromptT5(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PromptT5, self).__init__()

        try:
            if config.get("model","model_size")=="small":
                model = "t5-small"
                ckp = "T5SmallForMaskedLM"
                self.hidden_size = 512
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
                output = self.encoder(input_ids=data["inputx"], labels=data["label"])
                performance = kwargs["performance"]

                if int(kwargs["step"]%100) == 0:
                    gen = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=config.getint("eval","max_length"))

                    if "squad" in config.get("data","train_dataset_type") or "nq_open" in config.get("data","train_dataset_type") or "multi_news" in config.get("data","train_dataset_type") or "samsum" in config.get("data","train_dataset_type"):
                        performance = train_bleu(gen['sequences'], data["label"], config.get("data","train_dataset_type"))

                    else:
                        performance = train_acc(gen['sequences'], data["label"], config.get("data","train_dataset_type"))

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

            #for l in self.encoder.generate.state_dict():
            #    print(l)
            #print(kwargs)
            #exit()
            #{'args': Namespace(activate_neuron=True, checkpoint=None, comment=None, config='config/activate_neuronPromptT5.config', do_test=False, gpu='4', local_rank=-1, mode='valid', model_transfer_projector=False, pre_train_mlm=False, projector=None, prompt_emb_output=False, replacing_prompt='task_prompt_emb/IMDBPromptT5', save_name=None, seed=None, task_transfer_projector=False)}
            #print(data["inputx"])
            #print(data["inputx"].shape)
            #exit()

            output = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=config.getint("eval","max_length"))
            #output = self.encoder.generate(input_ids=data["inputx"], num_beams=config.getint("eval","num_beams"), output_scores=True, return_dict_in_generate=True, min_length=config.getint("eval","min_length"), max_length=2)


            #print("======")
            #print(output)
            #print("======")
            #exit()


            #generated_tokens = self.encoder.generate(input_ids=kwargs['input_ids'],attention_mask=kwargs['attention_mask'],num_beams=kwargs['num_beams'],output_scores=True,return_dict_in_generate=True)
            #generated_tokens = output['sequences']
            #print(generated_tokens)
            #print("-----")
            #print(generated_tokens.shape)
            #exit()

            #print(output["scores"])
            #print(output["scores"][0])
            #print(output["scores"][0].shape)
            #exit()




            if "squad" in config.get("data","train_dataset_type") or "nq_open" in config.get("data","train_dataset_type") or "multi_news" in config.get("data","train_dataset_type") or "samsum" in config.get("data","train_dataset_type"):
                acc_result = bleu(output['sequences'], data["label"], acc_result, config.get("data","train_dataset_type"))

            else:
                hidden_score = output["scores"][0]

                acc_result = acc(output['sequences'], data["label"], acc_result, config.get("data","train_dataset_type"), hidden_score=hidden_score)


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




def train_acc(score, label, dataset):
    '''
    if "nli" in dataset or "NLI" in dataset:
        #print(111111)
        #print(score)
        #print("----")
        #print(label)
        #print("======")
        #score = score[:,2:3]
        score = score[:,1:2]
        label = label[:,0:1]
    else:
    '''
    print()
    print(score)
    print("----")
    print(label)
    print("======")

    score = score[:,1:2]
    label = label[:,0:1]


    total = int(label.shape[0])
    right = int((score == label).int().sum())

    acc_result = round(float(right/total),4)

    return acc_result



def acc(score, label, acc_result, dataset, hidden_score=None):
    '''
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    if "nli" in dataset or "NLI" in dataset:
        #score = score[:,2:3]
        score = score[:,1:2]
        label = label[:,0:1]
    else:
        score = score[:,1:2]
        label = label[:,0:1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((score == label).int().sum())
    '''


    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}


    acc_result['total'] += int(label.shape[0])
    #print(label)
    #exit()
    label = label[:,0:1]

    if dataset == "IMDB":
        #negative: 2841, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==1465] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "SST2":
        #negative: 2841, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==1465] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "laptop":
        #negative: 2841, moderate:8107, positive:1465, conflict:4129
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,8107].unsqueeze(1), hidden_score[:,1465].unsqueeze(1), hidden_score[:,4129].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==8107] = 1
        label[label==1465] = 2
        label[label==4129] = 3
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "restaurant":
        #negative: 2841, moderate:8107, positive:1465, conflict:4129
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,8107].unsqueeze(1), hidden_score[:,1465].unsqueeze(1), hidden_score[:,4129].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==8107] = 1
        label[label==1465] = 2
        label[label==4129] = 3
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "movierationales":
        #negative: 2841, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==1465] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "tweetevalsentiment":
        #negative: 2841, moderate:8107, positive:1465
        score = torch.cat([hidden_score[:,2841].unsqueeze(1), hidden_score[:,8107].unsqueeze(1), hidden_score[:,1465].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==2841] = 0
        label[label==8107] = 1
        label[label==1465] = 2
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "MNLI":
        #contradiction: 27252, neutral: 7163, entailment: 3   #[3, 35, 5756, 297]
        score = torch.cat([hidden_score[:,27252].unsqueeze(1), hidden_score[:,7163].unsqueeze(1), hidden_score[:,3].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==27252] = 0
        label[label==7163] = 1
        label[label==3] = 2
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "QNLI":
        #contradiction: 27252, entailment: 3   #[3, 35, 5756, 297]
        score = torch.cat([hidden_score[:,27252].unsqueeze(1), hidden_score[:,3].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==27252] = 0
        label[label==3] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif dataset == "snli":
        #contradiction: 27252, neutral: 7163, entailment: 3   #[3, 35, 5756, 297]
        score = torch.cat([hidden_score[:,27252].unsqueeze(1), hidden_score[:,7163].unsqueeze(1), hidden_score[:,3].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==27252] = 0
        label[label==7163] = 1
        label[label==3] = 2
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif "ethics" in dataset:
        #unacceptable: 29452, acceptable: 9961
        score = torch.cat([hidden_score[:,29452].unsqueeze(1), hidden_score[:,9961].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==29452] = 0
        label[label==9961] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif "QQP" in dataset or "MRPC" in dataset:
        #false: 6136, true:1176
        score = torch.cat([hidden_score[:,6136].unsqueeze(1), hidden_score[:,1176].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==6136] = 0
        label[label==1176] = 1
        #label = label.reshape(int(label.shape[0]),int(label.shape[1]))
        label = label.reshape(int(label.shape[0]))
    elif "activate" in dataset:
        #false: 6136, true:1176
        score = torch.cat([hidden_score[:,6136].unsqueeze(1), hidden_score[:,1176].unsqueeze(1)], dim=1)
        score = torch.argmax(score, dim=1)
        label[label==6136] = 0
        label[label==1176] = 1
        label = label.reshape(int(label.shape[0]))
    else:
        print("Eval metrics wrong!!!")
        exit()



    #acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((score == label).int().sum())

    return acc_result





def train_bleu(score, label, dataset):
    total_bleu = 0
    length = len(label)

    #references = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    references = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    #hypotheses = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]
    hypotheses = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]

    total_bleu = 0
    for l in range(len(hypotheses)):
        print(references[l])
        #y = [references[l].lower().split()]
        y = [references[l]]
        print(hypotheses[l])
        #y_ = hypotheses[l].lower().split()
        y_ = hypotheses[l]
        print("-----")
        if len(y)!=0 and len(y_)!=0:
            #b = sentence_bleu(y, y_, weights=(0.35, 0.35, 0.15, 0.15), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
            b = sentence_bleu(y, y_, weights=(0.7, 0.3, 0.0, 0.0), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
        else:
            b = 0
        print(b)
        total_bleu+=b
    print("========")
    result = round(float(total_bleu/length),4)

    return result




def bleu(score, label, acc_result, dataset):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    acc_result['total'] += int(label.shape[0])

    #references = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    references = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    #hypotheses = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]
    hypotheses = [tokenizer.convert_ids_to_tokens(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]

    total_bleu = 0
    for l in range(len(hypotheses)):
        #print(references[l].lower().split())
        #y = [references[l].lower().split()]
        y = [references[l]]
        #print(hypotheses[l].lower().split())
        #y_ = hypotheses[l].lower().split()
        y_ = hypotheses[l]
        #print("-----")
        if len(y)!=0 and len(y_)!=0:
            b = sentence_bleu(y, y_, weights=(0.7, 0.3, 0.0, 0.0), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
            #b = sentence_bleu(y, y_, weights=(0.35, 0.35, 0.15, 0.15), smoothing_function=smoother.method1) #b-1, b-2, b-3, b-4
        else:
            b = 0
        #print(b)
        total_bleu+=b
    #print("========")

    '''
    references = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in label]
    hypotheses = [tokenizer.decode(l[l!=-100].tolist(), skip_special_tokens=True) for l in score]
    total_bleu = get_moses_multi_bleu(hypotheses, references, lowercase=True)
    if total_bleu == None:
        total_bleu = 0
    '''
    acc_result['right'] += int(total_bleu)

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
