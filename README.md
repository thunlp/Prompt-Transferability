
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
Imagine:

Task base:
(AE)
![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/SENTIMENT.jpg)
![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/NLI.jpg)


Domain base:
(AE)
![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain.jpg)


Demo random combination:
(AE)
![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/all_task.jpg)

Model transfer
![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/roberta_bert_prompt.jpg)

MLM task
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

https://arxiv.org/pdf/2102.12206.pdf
--
https://arxiv.org/pdf/2106.04489.pdf

1. Same task same domain: split dataset into two parts
2. Same task differnt domain: mlm, don't stop pre-training (general task), some glue task 
3. Smae domain different task: mlm, don't stop pre-training (general task)
4. task transfer: matrix task-task
5. model transfer: matrix bert-roberta
---
