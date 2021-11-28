import torch
from transformers import T5Tokenizer
#from transformers import T5ForConditionalGeneration

from transformers import AutoConfig
from modeling_t5 import T5ForConditionalGeneration
plmconfig = AutoConfig.from_pretrained('t5-base')
#print(plmconfig)
#exit()
plmconfig.prompt_num = 100
plmconfig.prompt_len = 100

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base', config=plmconfig)

#print("=====")
print(tokenizer("answer",add_special_tokens=False))
print(tokenizer("answer"))
exit()


'''
input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt', add_special_tokens=False).input_ids

labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the', return_tensors='pt', add_special_tokens=False).input_ids
'''

input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
print("-----")
print(input_ids)
input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
print("-----")
print(input_ids)
labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
print("-----")
print(labels)
labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt', add_special_tokens=True).input_ids
print("-----")
print(labels)
labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt', add_special_tokens=False).input_ids
print("-----")
print(labels)
#print(tokenizer.decode(labels[0]))

exit()


inputx = []
label = []

#a = list(range(-1,-100))
#a = list(range(1,100))
#print(a)
#exit()
prompt = []
for i in range(1,100):
    prompt.append(-i)
#print(prompt)
#exit()


for i in range(1):
    tokens = prompt + [1,2,3,4] + tokenizer.encode("<extra_id_0>", add_special_tokens=False)
    #tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))
    inputx.append(tokens)

    #target = tokenizer("<extra_id_0>"+" "+"negative", add_special_tokens=False).input_ids
    target = tokenizer("<extra_id_0>"+" "+"negative").input_ids
    label.append(target)

    ret = {
        "inputx": torch.tensor(inputx, dtype=torch.long),
        "target": torch.tensor(label, dtype=torch.long),
    }

#print(ret)
#exit()

input_ids = ret["inputx"]
labels = ret["target"]



print("=======")
print(input_ids)
print("-----")
print(labels)
print("=======")
exit()

outputs = model(input_ids=input_ids, labels=labels)

loss = outputs.loss
logits = outputs.logits

print(loss)
print(logits)
