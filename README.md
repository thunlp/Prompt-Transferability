
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

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_single_PCA_3D.jpg)-->

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_single_PCA_2D.jpg)-->



- Domain similiarty (Acc result): draw_cuda_mlmprompt_by_acc.py 

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_PCA_3D.jpg)-->

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_acc_PCA_2D.jpg)-->


- Domain similiarty (PCA) 

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_PCA_3D.jpg)-->

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain_sim_PCA_2D.jpg)-->

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
| -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- | -MLM- |
| IMDB_lm | 50.4 | 25.1 |  | 68.8 | 16.7 |  |  |  | 51.3 |  |  |
| laptop_lm | 50.4 | 5.2 |  | 67.7 | 8.3 |  |  |  | 51.7 |  |  |
| MRPC_lm | 50.4 | 2.3 |  | 68.8 | 1.2 |  |  |  | 51.2 |  |  |
| restaurant_lm | 50.3 | 13.4 |  | 68.2  | 15.1 |  |  |  | 51.4 |  |  |
| SST2_lm | 50.3 | 2.3 |  | 67.7 | 1.2 |  |  |  | 51.3 |  |  |


- Task similiarty (Replace prompt with various task-specific prompts and account acc.): draw_cuda_taskprompt_by_acc.py

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_acc_sim_3D.jpg)-->

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_acc_sim_2D.jpg)-->

- Task similiarty (AE) 

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_sim_AE.jpg)-->

- Task similiarty (PCA) 

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_sim_PCA_3D.jpg)-->

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/task_sim_PCA_2D.jpg)-->


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

##### Intrinsic Dimensionality

|   | IMDB | laptop | MNLI | MRPC | QNLI | QQP | restaurant | RTE | SST2 | STSB | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Non_Proj | 89 | 77.0 | 82.9 | 76.5 | 89.9 | 74.5 | 78.8 | 53.3 | 93.5 | | 42.1 |
| Proj(no:MRPC,QNLI,STSB)_76800_to_1200 | 87.9 | 73.0 | 47.6 | 39.1 | 53.5 | 64.9 | 80.5 | 53.1 | 91.5 | | 39.1 |
| Proj(no:STSB)_76800_to_768 |  |  |  |  |  |  |  |  |  | |  |
| Proj(no:STSB)_76800_to_3 | 88.2 | 72.5 | 53.4 | 84.1 | 81.8 | 66.5 | 77.2 | 59.0 | 92.1 |  | 26.6 |


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
- Can work: Need to change AE in init_tool.py for different setting 

|   | IMDB(2)| laptop(4)| MNLI(3)| MRPC(2)| QNLI(2)| QQP(2)| restaurant(4)| RTE(2)| SST2(2)| STSB | WNLI(2)|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Non_Proj | 50.4 | 15.2 | 35.5 | 69.3 | 49.3 | 36.8 | 2.8 | 50.4 | 51.2 | | 53.1 |
| Proj (trained with mlm) | 58.4 | 49.4 | 35.5 | 31.0 | 50.5 | 63.2 | 63.1 | 47.3 | 54.1 |  | 48.4 |
| Proj (trained with task (no STSB)) | 87.5 | 73.3 | 52.7 | 90.9 | 80.9 | 70.5 | 79.5 | 50.8 | 92.2 |  | 15.6 |
| Proj (trained with imdb,laptop) | 86.8 | 70.9 |  |  |  |  | 75.8 |  | 88.5 |  |  |




- Revised:
* replace predict token with the [mask] for Roberta, Bert
* re-train all task prompt : mlm, all task (restaurant, laptop: done)
* re-train all transfer matrix
* project (task transfer) --> reduce to 3 dim


* mlm_cross, cross, project --> didn't match the key
* cross, project --> didn't match roberta or bert prompt
* Re-train transfer-matric (task, model, mlm[Done])
* mlm need to train till 32 epoch

- To do:
* Measure MLM prompt similiarty with other tasks
* Bert mlm promt: haven't trained del_sst2_lm
* MLM task: using prompt and without prompt
* Fine-tuned model transfer
* Change training code: only save prompt emb instead of whole model
-Emergency: 1. mlm prompt transfer 2. mlm prompt for cross model


* Re-training: 
1. task prompt
2. mlm prompt
3. extract task prompt
4. projector_task
5. projector_mlm
6. cross_task
7. cross_mlm


* Add dataset
1. snli
2. anli
3. recast
4. yelp_polarity (Alter to smaller epoches)
5. pragmeval

* Ready to train
- NLI
1. snli
2. anli (r_3)
3. recast_factuality

- Sentiment
1. tweeteval/sentiment
3. movie_rationales/default

- Discourse
1. pragmeval/emobank-arousal, emobankarousal
2. pragmeval/persuasiveness-relevance, persuasivenessrelevance
3. pragmeval/persuasiveness-specificity, persuasivenessspecificity
4. pragmeval/emobank-dominance, emobankdominance
5. pragmeval/squinky-implicature, squinkyimplicature
6. pragmeval/squinky-formality, squinkyformality

# Need to revised PromptRoberta, PromptBert --> add low, high tokens


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

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/SENTIMENT.jpg)-->
<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/NLI.jpg)-->
---


- Domain base: (AE)

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/domain.jpg)-->
---


- Demo random combination: (AE)

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/all_task.jpg)-->
---


- Model transfer

<!--![](https://github.com/yushengsu-thu/prompt/blob/main/exp_results/roberta_bert_prompt.jpg)-->
---


- MLM task

<!-- ![](https://github.com/yushengsu-thu/prompt/blob/main/output.jpg) -->
---


- output figure
<!--![](https://github.com/yushengsu-thu/prompt/blob/main/output.jpg)-->


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
3. News: agnews_s1, agnews_s2 ([World, Sports, Business, Sci/Tech]) 
---
- Sentimentation Classification (Remove label bias: remove the min label and blance the negative and postive sample)
---
- Task and Domain and label: split all dataset into two parts: D_s1 and D_s2
---
