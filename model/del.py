
'''
from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from modelling_bert import BertForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = "bert-base-uncased"

plmconfig = AutoConfig.from_pretrained(model)
config = AutoConfig.from_pretrained(model)
#print(config)
#exit()
encoder = BertForMaskedLM.from_pretrained(model, config=plmconfig)

print(encoder)
#print(encoder["BertOnlyMLMHead"])
#print([param for param in model.bert.parameters()])
'''

from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer
from modelling_roberta import RobertaForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

model = "roberta-base"

plmconfig = AutoConfig.from_pretrained(model)
config = AutoConfig.from_pretrained(model)
#print(config)
#exit()
encoder = RobertaForMaskedLM.from_pretrained(model, config=plmconfig)

print(encoder)
