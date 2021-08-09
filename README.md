
# Prompt Analysis


![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)

![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg)


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
### Domain Similarity
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



### Domain Prompt Transferability

| : Filename(dataset) ; __ : prompt_emb

|   | SST-2_s1 | SST-2_s2 | IMDB_s1 | IMDB_s2 | cs_wiki_s1 | cs_wiki_s2 | scierc_s1 | scierc_s2 | agnews_s1 | agnews_s2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SST-2_s1 | 68.9 | 68.6 | 65.4 | 65.4 | 46.5 | 46.3 | 52.3 | 50.9 | 64.0 | 64.5 |
| SST-2_s2 | 67.7 | 67.5 | 65.2 | 65.2 | 46.9 | 47.4 | 50.4 | 52.5 | 63.3 | 63.7 |
| IMDB_s1 | 56.7 | 56.9 | 72.0 | 72.1 | 47.4 | 47.0 | 49.2 | 51.1 | 69.4 | 70.3 |
| IMDB_s2 | 55.3 | 56.3 | 72.0 | 71.9 | 47.3 | 47.1 | 48.5 | 49.4 | 69.4 | 68.9 |
| cs_wiki_s1 | 42.7 | 43.2 | 62.9 | 64.3 | 51.7 | 51.0 | 39.6 | 38.9 | 57.3 | 57.1 |
| cs_wiki_s2 | 44.5 | 46.1 | 62.9 | 65.5 | 51.1 | 51.0 | 39.9 | 40.3 | 57.5 | 57.7 |
| scierc_s1 | 49.2 | 50.8 | 64.5 | 64.3 | 45.1 | 45.0 | 69.4 | 70.0 | 60.4 | 60.6 |
| scierc_s2 | 55.2 | 54.9 | 65.4 | 65.5 | 45.1 | 45.5 | 68.3 | 67.3 | 62.3 | 62.2 |
| agnews_s1 | 54.9 | 55.6 | 69.0 | 68.9 | 46.8 | 47.1 | 51.1 | 50.7 | 74.5 | 74.5 |
| agnews_s2 | 54.6 | 56.1 | 68.9 | 68.9 | 47.1 | 46.2 | 49.6 | 51.3 | 74.0 | 74.5 |


- Domain similiarty (Acc result single): draw_cuda_mlmprompt_by_acc.py 

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_single_PCA_3D.jpg)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_single_PCA_2D.jpg)



- Domain similiarty (Acc result): draw_cuda_mlmprompt_by_acc.py 

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_PCA_3D.jpg)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_PCA_2D.jpg)


- Domain similiarty (PCA) 

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_PCA_3D.jpg)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_PCA_2D.jpg)

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
| IMDB | | | | | | | | | | | |
| laptop | | | | | | | | | | | |
| MNLI | | | | | | | | | | | |
| MRPC | | | | | | | | | | | |
| QNLI | | | | | | | | | | | |
| QQP | | | | | | | | | | | |
| restaurant | | | | | | | | | | | |
| RTE | | | | | | | | | | | |
| SST2 | | | | | | | | | | | |
| STSB | | | | | | | | | | | |
| WNLI | | | | | | | | | | | |

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


- Task similiarty (Replace prompt with various task-specific prompts and account acc.): draw_cuda_taskprompt_by_acc.py

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_acc_sim_3D.jpg)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_acc_sim_2D.jpg)

- Task similiarty (AE) 

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_sim_AE.jpg)

- Task similiarty (PCA) 

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_sim_PCA_3D.jpg)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_sim_PCA_2D.jpg)


- Bert

|   | IMDB | laptop | MNLI | MRPC | QNLI | QQP | restaurant | RTE | SST2 | STSB | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMDB | | | | | | | | | | | |
| laptop | | | | | | | | | | | |
| MNLI | | | | | | | | | | | |
| MRPC | | | | | | | | | | | |
| QNLI | | | | | | | | | | | |
| QQP | | | | | | | | | | | |
| restaurant | | | | | | | | | | | |
| RTE | | | | | | | | | | | |
| SST2 | | | | | | | | | | | |
| STSB | | | | | | | | | | | |
| WNLI | | | | | | | | | | | |


### Projector

(-): Have training instances

|   | IMDB | laptop | MNLI | MRPC(-) | QNLI(-) | QQP | restaurant | RTE | SST2 | STSB(-) | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Non_Proj | 89 | 77.0 | 82.9 | 76.5 | 89.9 | 74.5 | 72.1 | 53.3 | 93.5 | | 42.1 |
| Proj | 87.9 | 73.0 | 47.6 | 39.1 | 53.5 | 64.9 | 80.5 | 53.1 | 91.5 | | 39.1 |


---
## Model Transferability

### Task Similarity
Model:
- Roberta
- Bert

### Non-projector
- Roberta

- Bert

- Didn't work

### Projector
- Can work 












---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
Imagine:

- Task base: (AE)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/SENTIMENT.jpg)
![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/NLI.jpg)
---


- Domain base: (AE)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain.jpg)
---


- Demo random combination: (AE)

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/all_task.jpg)
---


- Model transfer

![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/roberta_bert_prompt.jpg)
---


- MLM task

<!-- ![](https://github.com/yushengsu-thu/prompt/blob/main/output.jpg) -->
---


- output figure
![](https://github.com/yushengsu-thu/prompt/blob/main/output.jpg)


---
Finding:
- 1. [exp_results/exp_optimize_prompt_to_task_direction.txt]: Prompt emb can be seperated by prompt tuning. (Observe epochs from 1 to 15)
- 2. [exp_results/exp_task_radomseed.txt]: Prompt emb in different random seed 

- Conclusion: Apply EuclideanDistances to measures similiarty is more obvious. 

---
Performance:
- laptop: 64.5%
- restaurant: 69.7% 
- SST2: 93.0%
- QQP: 74%

---
- Similiarty:

Table:

---
To-Do:
- Read dataset:
Same task, same domain: split a dataset into two parts.
Same task, Different domain: https://arxiv.org/pdf/2102.12206.pdf
Different Task, Same Domain:

clear distribution (dataset): https://arxiv.org/pdf/2106.04489.pdf

!!!!!!!!! Re-range the order of the tensor: (and the corresponding to dots on the figures )

1. Try latest trained Projector
2. project bert prompt and roberta prompt
3. use bert prompt for roberta model (transfer)
4. Observe: the similiar tasks have the common cluster (the task should be far different)
5. Fix projector_prompt.py --> I can just change prompt emb in the init_tool.py. (Think: how to not use prompt_output parameter)
6. create.py (extract prompt....think better methods)
7. Fix code (valid_projector_prompt.py) like (valid_roberta-bert_prompt.py)
8. To-Do: merge Bert and Roberta in Model and Formatter
9. Merge: (train.py, train_lm.py), (valid.py, valid_lm.py), (mlmPrompt, PromptRoberta, PromptBert)
10. MUST merge model/--Prompt--.py. Prompt_mlm is the optimal.
11. prompt_emt: using m-way-k-shot training
12. -Use trained AE (task_transfer) for restaurantPromptRoberta
---
Extra dataset:
--
- https://arxiv.org/pdf/2102.12206.pdf

- https://arxiv.org/pdf/2106.04489.pdf

- https://arxiv.org/pdf/2004.10964.pdf

- CS-wiki: git@github.com:pmernyei/wiki-cs-dataset.git

1. Same task same domain: split dataset into two parts
2. Same task differnt domain: mlm, don't stop pre-training (general task), some glue task 
3. Smae domain different task: mlm, don't stop pre-training (general task)
3'. domain: only do mlm on differet domain datasets
4. task transfer: matrix task-task
5. model transfer: matrix bert-roberta
---
- Same Domain different Task:
- MLM
1. Movie review: SST-2_s1, SST-2_s2, IMDB_s1, IMDB_s2
2. CS: cs_wiki_s1, cs_wiki_s2, scierc 
3. News: agnews_s1, agnews_s2 
---
- Sentimentation Classification (Remove label bias: remove the min label and blance the negative and postive sample)
---
- Task and Domain and label: split all dataset into two parts: D_s1 and D_s2
---
