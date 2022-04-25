
# Prompt Analysis


Note
=============
#### Environment Setup
```
bash requirement.sh
```


#### Training
```
bash train.sh
```


#### Valid
```
bash valid.sh
```


#### Extract Prompt Embedding
```
bash create_prompt_emb.sh
```

### Rank Similiarty
```
rank_task_sim.py
```

### train.py
```
Generate prompt for each task
```

### train projector.py
```
Train an AE
```

### valid projector prompt.py
```
Compare original prompt with compressed prompt
```




To-Do
=============
Model Size and Type

|   |  BASE | LARGE  |
| ------------ | ------------ | ------------ |
| RoBERTa  |  Done  |  Done  |
| T-5  | -  |  - |
| GPT  |  - | -  |   |


---

Downstream Dataset
```
- GLUE:  MNLI, QNLI, MRPC, QQP, RTE, SST2, WNLI, STSB, ~~CoLA~~
- QA: ~~SQUAD~~
- RE: Fewrel
- GCAE: laptop, restaurant
- IMDB
```

Task:
- Sentiment Classification: SST2, laptop, restaurant, IMDB
- Paraphrase: MRPC, QQP, STSB(?)
- NLI: MNLI, RTE, WNLI(Coreference), QNLI(QA)
- RE: Fewrel
- Sentence Similiarity: STSB
- QA: QNLI(QA), SQUAD


Domain:
- misc.: STSB, MNLI
- movie: SST2, IMDB
- news: MRPC, RTE
- social question: QQP
- Wiki: QNLI, RE, RTE
- Fiction: WNLI
- reataurant: restaurant
- computer: laptop

---

## Task Transferability

---
### Task Similarity

Task:
- Sentiment Classification: SST2, laptop, restaurant, IMDB
- Paraphrase: MRPC, QQP, STSB(?)
- NLI: MNLI, RTE, WNLI(Coreference), QNLI(QA)
- RE: Fewrel
- Sentence Similiarity: STSB
- QA: QNLI(QA), SQUAD


| Use label similiarty 

|   | IMDB | laptop | MNLI | MRPC | QNLI | QQP | restaurant | RTE | SST2 | STSB | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### Task Prompt Transferability

| : Filename(dataset) ; __ : prompt_emb

### Non-projector


---
## Model Transferability

### Task Similarity
Model:
- Roberta
- Bert



---
---
---
=============
# 2021.09.09

### Train bert-medium task prompt
- Need to add datasetname to train_bert_medium_sample.sh and create data-specific config files.
```
bash train_bert_medium_sample.sh
```

### Generate task-specific prompt and save to task_prompt_emb 
```
python3 copy_prompt_to_taskpromptdir_medium.py
```


### Train task projector (with Task prompt)
- Edit crossPromptBertMedium.config
```
bash train_bert_cross_medium.sh
```







### Train bert-medium mlm prompt
- Need to add datasetname to train_lm_s_bert_medium.sh and create data-specific config files.
```
bash train_lm_s_bert_medium.sh
```


### Generate task-specific prompt and save to task_prompt_emb 
```
python3 copy_prompt_to_taskpromptdir_medium.py
```

### Train mlm projector (with mlm prompt)
- Edit cross_mlmPromptBertMedium.config
```
bash train_mlm_medium_cross.sh
```

