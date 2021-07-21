
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
To Do:
!!!!!!!!! Rerange the order of the tensor: (and the corresponding fot on the figures)

1. Try latest trained Projector
2. project bert prompt and roberta prompt
3. use bert prompt for roberta model (transfer)
4. Observe: the similiar tasks have the common cluster (the task should be far different)
5. Fix projector_prompt.py --> I can just change prompt emb in the init_tool.py.
---
![](https://github.com/yushengsu-thu/prompt/blob/main/roberta_bert_prompt.jpg)
