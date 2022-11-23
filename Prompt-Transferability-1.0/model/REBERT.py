import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoModel,AutoModelForMaskedLM,AutoTokenizer

class REBERT(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(REBERT, self).__init__()

        self.encoder = AutoModel.from_pretrained("roberta-base")
        self.hidden_size = 768
        self.label_num = config.getint("train", "label_num")
        self.fc = nn.Linear(self.hidden_size * 2, self.label_num)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, config, gpu_list, acc_result, mode):
        batch = data["inputx"].shape[0]
        output = self.encoder(data["inputx"], attention_mask=data['mask'])
        hiddens = output["last_hidden_state"]
        arange = torch.arange(batch).to(hiddens.device)
        head_rep = hiddens[arange, data["headpos"]]
        tail_rep = hiddens[arange, data["tailpos"]]
        features = torch.cat([head_rep, tail_rep], dim = 1)
        score = self.fc(features)
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
