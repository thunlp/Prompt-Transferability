
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


## Domain Similarity
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


__ : dataset, | : prompt

|   | SST-2_s1 | SST-2_s2 | IMDB_s1 | IMDB_s2 | cs_wiki_s1 | cs_wiki_s2 | scierc_s1 | scierc_s2 | agnews_s1 | agnews_s2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SST-2_s1 | | | | | 46.5 | 46.3 | | | 63.7 | 64.5 |
| SST-2_s2 | | | | | 46.9 | 47.4 | | | 63.8 | 63.7 |
| IMDB_s1 | | | | | 47.4 | 47.0 | | | 69.1 | 70.3 |
| IMDB_s2 | | | | | 47.3 | 47.1 | | | 69.6 | 68.9 |
| cs_wiki_s1 | | | | | 51.7 | 51.0 | | | 57.4 | 57.1 |
| cs_wiki_s2 | | | | | 51.1 | 51.0 | | | 57.7 | 57.7 |
| scierc_s1 | | | | | 45.1 | 45.0 | | | 60.4 | 60.6 |
| scierc_s2 | | | | | 45.1 | 45.5 | | | 62.3 | 62.2 |
| agnews_s1 | | | | | 46.8 | 47.1 | | | 74.6 | 74.5 |
| agnews_s2 | | | | | 47.1 | 46.2 | | | 74.3 | 74.5 |




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

![](https://github.com/yushengsu-thu/prompt/blob/main/output.jpg)
---



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
