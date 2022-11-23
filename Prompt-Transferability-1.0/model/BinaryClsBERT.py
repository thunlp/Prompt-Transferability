import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoModel,AutoModelForMaskedLM,AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
try:
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
except:
    tokenizer = AutoTokenizer.from_pretrained("RobertaForMaskedLM/roberta-base")


class PromptClsBERT(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PromptClsBERT, self).__init__()

        self.encoder = AutoModelForMaskedLM.from_pretrained("roberta-base")
        self.hidden_size = 768
        # self.fc = nn.Linear(self.hidden_size, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.prompt_num = config.getint("prompt", "prompt_len") # + 1
        self.init_prompt_emb()
        # self.prompt_label = nn.Parameter(torch.Tensor(2, self.hidden_size), requires_grad=True)
        self.prompt_label = nn.Linear(2, self.hidden_size, bias=False)
        self.prompt_label.weight.data = self.encoder.lm_head.decoder.weight[[10932, 2362]]
        self.temperature = 2
        print("init the model PromptClsBERT done")

    def lower_temp(self, ratio):
        self.temperature *= ratio

    def init_prompt_emb(self):

        init_ids = [] #tokenizer.encode("the relation between the first sentence and the second sentence is")
        pad_num = self.prompt_num - len(init_ids)
        init_ids.extend([tokenizer.mask_token_id] * pad_num)
        self.prompt_emb = nn.Embedding(self.prompt_num, self.hidden_size).from_pretrained(self.encoder.get_input_embeddings()(torch.tensor(init_ids, dtype=torch.long)), freeze=False)
        self.class_token_id = torch.tensor([10932, 2362])

    def gen_attention_score(self, logits):
        # self.prompt_label: 2, hidden_size
        # self.encoder.lm_head.decoder.weight: vocab_size, hidden_size
        # logits: batch, vocab_size
        att = torch.softmax(torch.mm(self.prompt_label.weight, torch.transpose(self.encoder.lm_head.decoder.weight, 0, 1) / self.temperature), dim = 1) # 2, vocab_size
        score = torch.mm(logits, torch.transpose(att, 0, 1))
        return score

    def forward(self, data, config, gpu_list, acc_result, mode):
        # print(self.prompt_label)
        # print(data["inputx"][0].tolist())
        batch, seq_len = data["inputx"].shape[0], data["inputx"].shape[1]
        prompt = self.prompt_emb.weight # prompt_len, 768

        input = self.encoder.get_input_embeddings()(data["inputx"])
        embs = torch.cat([prompt.unsqueeze(0).repeat(batch, 1, 1), input], dim = 1)
        # print(embs[0][:10, :10].tolist())
        # print("=" * 20)
        output = self.encoder(attention_mask=data['mask'], inputs_embeds=embs)
        logits = output["logits"] # batch, seq_len, vocab_size
        mask_logits = logits[:, 0] # batch, vocab_size
        score = self.gen_attention_score(mask_logits)
        # score = torch.cat([mask_logits[:, 10932].unsqueeze(1), mask_logits[:, 2362].unsqueeze(1)], dim = 1)

        loss = self.criterion(score, data["label"])
        acc_result = acc(score, data['label'], acc_result)

        return {'loss': loss, 'acc_result': acc_result}

class BinaryClsBERT(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BinaryClsBERT, self).__init__()

        self.encoder = AutoModel.from_pretrained("roberta-base")
        self.hidden_size = 768
        self.label_num = 2
        try:
            self.label_num = config.getint("train", "label_num")
        except:
            pass
        self.fc = nn.Linear(self.hidden_size, self.label_num)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, config, gpu_list, acc_result, mode):
        output = self.encoder(data["inputx"], attention_mask=data['mask'])
        score = self.fc(output['pooler_output']) # batch, class_num
        loss = self.criterion(score, data["label"])
        acc_result = acc(score, data['label'], acc_result)

        return {'loss': loss, 'acc_result': acc_result}

def acc(score, label, acc_result):
    if acc_result is None:
        acc_result = {'total': 0, 'right': 0}
    predict = torch.max(score, dim = 1)[1]
    acc_result['total'] += int(label.shape[0])
    acc_result['right'] += int((predict == label).int().sum())
    return acc_result
