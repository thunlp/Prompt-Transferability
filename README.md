
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
Comment out text in markdown: https://stackoverflow.com/questions/16525877/how-do-you-display-a-section-of-plain-text-in-github-markdown


## Domain and Task Transferability

---
### Domain Similarity [DONE]
- Top 5000 token
- 20000 instances for Each dataset


|   | SST-2_s1 | SST-2_s2 | IMDB_s1 | IMDB_s2 | cs_wiki_s1 | cs_wiki_s2 | scierc_s1 | scierc_s2 | agnews_s1 | agnews_s2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SST-2_s1 | 100.0% | 100.0% | 56.8% | 56.8% | 39.2% | 39.1% | 30.4% | 30.2% | 34.4% | 34.5% |
| SST-2_s2 | 100.0% | 100.0% | 56.8% | 56.8% | 39.2% | 39.1% | 30.4% | 30.2% | 34.4% | 34.5% |
| IMDB_s1 | 56.8% | 56.8% | 100.0% | 100.0% | 40.7% | 40.6% | 33.5% | 33.3% | 48.5% | 48.8% |
| IMDB_s2 | 56.8% | 56.8% | 100.0% | 100.0% | 40.7% | 40.6% | 33.5% | 33.3% | 48.5% | 48.8% |
| cs_wiki_s1 | 39.2% | 39.2% | 40.7% | 40.7% | 100.0% | 99.1% | 45.0% | 44.8% | 43.3% | 43.6% |
| cs_wiki_s2 | 39.1% | 39.1% | 40.6% | 40.6% | 99.1% | 100.0% | 44.9% | 44.8% | 43.3% | 43.4% |
| scierc_s1 | 30.4% | 30.4% | 33.5% | 33.5% | 45.0% | 44.9% | 100.0% | 98.3% | 32.6% | 32.5% |
| scierc_s2 | 30.2% | 30.2% | 33.3% | 33.3% | 44.8% | 44.8% | 98.3% | 100.0% | 32.6% | 32.5% |
| agnews_s1 | 34.4% | 34.4% | 48.5% | 48.5% | 43.5% | 43.3% | 32.6% | 32.6% | 100.0% | 90.0% |
| agnews_s2 | 34.5% | 34.5% | 48.8% | 48.8% | 43.6% | 43.4% | 32.5% | 32.5% | 90.0% | 100.0% |

Code: split_dataset.py, train_lm_s.sh, draw_cuda_mlmprompt_split.sh, valid_lm_replaceprompt.sh



### Domain Prompt Transferability [DONE]

| : Filename(dataset) ; __ : prompt_emb

|   | SST-2_s1 | SST-2_s2 | IMDB_s1 | IMDB_s2 | cs_wiki_s1 | cs_wiki_s2 | scierc_s1 | scierc_s2 | agnews_s1 | agnews_s2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SST-2_s1 | 68.2 | 68.3 | 64.4 | 64.2 | 47.1 | 47.9 | 51.8 | 53.7 | 64.0 | 63.9 |
| SST-2_s2 | 66.8 | 68.3 | 65.4 | 65.5 | 47.9 | 47.3 | 51.9 | 51.8 | 63.5 | 64.0 |
| IMDB_s1 | 57.2 | 58.6 | 72.0 | 72.0 | 46.8 | 47.0 | 49.8 | 50.3 | 69.8 | 72.0 |
| IMDB_s2 | 56.6 | 58.0 | 71.8 | 72.3 | 47.8 | 47.5 | 49.7 | 50.3 | 69.3 | 71.8 |
| cs_wiki_s1 | 44.7 | 43.3 | 61.8 | 61.8 | 50.8 | 51.7 | 39.6 | 38.9 | 57.0 | 57.0 |
| cs_wiki_s2 | 44.0 | 45.0 | 62.8 | 62.7 | 51.8 | 51.8 | 41.3 | 41.9 | 55.8 | 56.6 |
| scierc_s1 | 56.1 | 57.5 | 64.4 | 64.4 | 45.6 | 46.3 | 69.3 | 70.2 | 61.9 | 62.7 |
| scierc_s2 | 54.0 | 55.4 | 64.6 | 64.7 | 45.4  | 45.9 | 71.2 | 69.3 | 61.5 | 61.8 |
| agnews_s1 | 55.1 | 56.3 | 69.1 | 69.1 | 47.5 | 46.9 | 51.2 | 49.9 | 75.1 | 75.1 |
| agnews_s2 | 56.2 | 56.9 | 69.0 | 68.9 | 47.9 | 47.7 | 51.0 | 51.1 | 74.8 | 74.6 |


* Draw: mlm prompt can be split clearly by the boundary (Bert and Roberta) --> Learn a mapping: can work!

- Domain similiarty (Acc result single): draw_cuda_mlmprompt_by_acc.py 


<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_PCA_3D.jpg)-->


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

- Roberta

|   | IMDB | laptop | MNLI | MRPC | QNLI | QQP | restaurant | RTE | SST2 | STSB | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMDB | 89.0 | 28.3 | 37.4 | 69.0 | 51.2 | 36.8 | 25.8 | 51.6 | 84.7 | | 50.0 |
| laptop | 76.2 | 69.5 | 39.6 | 67.5 | 51.5 | 37.2 | 76.2 | 50.0 | 83.4 | | 53.1 |
| MNLI | 50.4 | 4.2 | 82.9 | 71.4 | 74.8 | 36.8 | 5.8 | 49.2 | 55.1 | | 57.8 |
| MRPC | 50.3 | 2.5 | 38.5 | 76.8 | 53.1 | 38.3 | 1.8 | 52.3 | 50.4 | | 46.9 |
| QNLI | 50.4 | 2.5 | 47.8 | 69.0 | 89.9 | 36.8 | 1.1 | 53.9 | 51.4 | | 51.6 |
| QQP | 50.4 | 2.5 | 33.1 | 32.2 | 50.4 | 74.5 | 1.2 | 48.8 | 51.1 | | 53.1 |
| restaurant | 74.5 | 65.6 | 37.6 | 67.5 | 50.6 | 36.8 | 77.1 | 54.7 | 84.4 | | 59.4 |
| RTE | 50.3 | 2.3 | 37.0 | 67.7 | 53.5 | 36.8 | 1.1 | 54.3 | 50.8 | | 50.0 |
| SST2 | 85.4 | 49.8 | 43.0 | 67.2 | 52.6 | 36.9 | 64.3 | 52.0 | 93.6 | | 46.9 |
| STSB | 50.3 | 2.5 | 37.2 | 68.2 | 54.5 | 36.8 | 1.3 | 52.0 | 50.0 | | 40.6 |
| WNLI | 50.3 | 21.1 | 36.4 | 68.0 | 49.6 | 36.8 | 17.0 | 52.7 | 50.7 | | 43.8 |
| -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- |
| IMDB_lm | 50.4 | 25.1 |  | 68.8 | 16.7 |  |  |  | 51.3 |  |  |
| laptop_lm | 50.4 | 5.2 |  | 67.7 | 8.3 |  |  |  | 51.7 |  |  |
| MRPC_lm | 50.4 | 2.3 |  | 68.8 | 1.2 |  |  |  | 51.2 |  |  |
| restaurant_lm | 50.3 | 13.4 |  | 68.2  | 15.1 |  |  |  | 51.4 |  |  |
| SST2_lm | 50.3 | 2.3 |  | 67.7 | 1.2 |  |  |  | 51.3 |  |  |


- Task similiarty (Replace prompt with various task-specific prompts and account acc.): draw_cuda_taskprompt_by_acc.py




- Bert

|   | IMDB | laptop | MNLI | MRPC | QNLI | QQP | restaurant | RTE | SST2 | STSB | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


### Projector

##### Intrinsic Dimensionality

|   | IMDB | laptop | MNLI | MRPC | QNLI | QQP | restaurant | RTE | SST2 | STSB | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Non_Proj | 89 | 77.0 | 82.9 | 76.5 | 89.9 | 74.5 | 78.8 | 53.3 | 93.5 | | 42.1 |
| Proj(no:MRPC,QNLI,STSB)_76800_to_1200 | 87.9 | 73.0 | 47.6 | 39.1 | 53.5 | 64.9 | 80.5 | 53.1 | 91.5 |  | 39.1 |
| Proj(no:STSB)_76800_to_768 |  |  |  |  |  |  |  |  |  |  |  |
| Proj(no:STSB)_76800_to_3 | 88.2 | 72.5 | 53.4 | 84.1 | 81.8 | 66.5 | 77.2 | 59.0 | 92.1 |  | 26.6 |


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

