# Prompt Analysis

![](https://pandao.github.io/editor.md/images/logos/editormd-logo-180x180.png)

![](https://img.shields.io/github/stars/pandao/editor.md.svg) ![](https://img.shields.io/github/forks/pandao/editor.md.svg) ![](https://img.shields.io/github/tag/pandao/editor.md.svg) ![](https://img.shields.io/github/release/pandao/editor.md.svg) ![](https://img.shields.io/github/issues/pandao/editor.md.svg) ![](https://img.shields.io/bower/v/editor.md.svg)


Note
=============
####Environment Setup
	bash requirement.sh



####Training
	bash train.sh



####Extract Prompt Embedding
	bash create_prompt_emb.sh


To-Do
=============
1. Model Size and Type

|   |  BASE | LARGE  |
| ------------ | ------------ | ------------ |
| RoBERTa  |  Done  |  Done  |
| T-5  | -  |  - |
| GPT  |  - | -  |   |




3. Downstream Dataset

GLUE:  MNLI, QNLI, MRPC, QQP, RTE, SST2, WNLI, STSB, ~~CoLA~~
QA: ~~SQUAD~~
RE: Fewrel

- Task
	Sentiment Classification: SST2
	Paraphrase: MRPC,  QQP
	Sentence Similiarity: STSB
	NLI: MNLI, QNLI(QA), RTE, WNLI(Coreference)
	RE: Fewrel

- Domain
	misc.: STSB, MNLI
	movie: SST2
	news: MRPC, RTE
	social question: QQP
	Wiki: QNLI, Fewrel, RTE
	Fiction: WNLI

- Similiarty:

Table:

|     | sst2 | rte | re | MNLI | MRPC | QNLI | QQP | WNLI | STSB
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| sst2  | -  | 0.66  | 0.06  | 0.04  | -0.01  | 0.02 | 0.02  | 0.00  | -0.00 |
| rte | 0.66  | -  | 0.07 | 0.05 | -0.02 | 0.02 | 0.01 | -0.00 | -0.02 |
| re  | 0.06  | 0.07 | -  | 0.02  | -0.01 | 0.01  | 0.02  | 0.02  | -0.01 |
| MNLI  | 0.04  | 0.05  |  0.02 | -  | -0.00  | 0.02  | 0.03  | 0.00  | 0.01 |
| MRPC  | -0.01  | -0.02  | -0.01  | 0.03  | -  | 0.02  | 0.01  | 0.03 | 0.01 |
| QNLI  | 0.02  | 0.02  | 0.01  | 0.02 | 0.02 |  - |  0.02 | 0.02  | 0.02  |
| QQP  |  0.02 |  0.01 | 0.01  | 0.03  | 0.01  | 0.02  |  - |  -0.0 | 0.01  |
| WNLI  |  0.00 | -0.00  | -0.00  |  - | 0.03  | 0.02  | -0.00  | -  | 0.02  |
| STSB  |  -0.00 |  -0.01 |  -0.01 | 0.01  |  0.01 |  0.02 |  0.01 |  0.03 | -  |




List:

---
sst2
CosineSimilarity:
rte 0.6607216000556946
re 0.062042467296123505
MNLI 0.044787678867578506
QQP 0.02038535475730896
QNLI 0.01816180720925331
WNLI 0.0019883119966834784
STSB -0.003573349444195628
MRPC -0.01410999707877636

EuclideanDistances:
rte 20.667238235473633
WNLI 27.981159210205078
MRPC 28.257226943969727
STSB 28.38043975830078
QNLI 33.33351516723633
QQP 46.172767639160156
MNLI 46.533233642578125
re 56.5198860168457



---
rte
CosineSimilarity:
sst2 0.6607216000556946
re 0.07017217576503754
MNLI 0.046767402440309525
QNLI 0.022116217762231827
QQP 0.014439152553677559
WNLI -0.0022752464283257723
MRPC -0.015595166012644768
STSB -0.01770864985883236

EuclideanDistances:
sst2 20.667238235473633
WNLI 21.278451919555664
MRPC 21.591501235961914
STSB 21.863563537597656
QNLI 27.955751419067383
QQP 42.66611862182617
MNLI 43.06961441040039
re 53.7603874206543



---
re
CosineSimilarity:
rte 0.07017217576503754
sst2 0.062042467296123505
MNLI 0.017445003613829613
QQP 0.014833556488156319
QNLI 0.005663424264639616
WNLI -0.002854109276086092
STSB -0.011063598096370697
MRPC -0.011364794336259365

EuclideanDistances:
WNLI 51.545379638671875
MRPC 51.698219299316406
STSB 51.80781173706055
rte 53.7603874206543
QNLI 54.68842315673828
sst2 56.5198860168457
QQP 63.15311050415039
MNLI 63.740089416503906



---
MNLI
CosineSimilarity:
rte 0.046767402440309525
sst2 0.044787678867578506
QQP 0.02906796522438526
QNLI 0.022226665169000626
re 0.017445003613829613
STSB 0.006048301700502634
WNLI 0.003950587939471006
MRPC -0.002348720794543624

EuclideanDistances:
WNLI 39.31630325317383
MRPC 39.47811508178711
STSB 39.55362319946289
rte 43.06961441040039
QNLI 43.1364860534668
sst2 46.533233642578125
QQP 53.41044235229492
re 63.740089416503906



---
MRPC
CosineSimilarity:
WNLI 0.028840098530054092
QNLI 0.016551252454519272
QQP 0.012579187750816345
STSB 0.006784915458410978
MNLI -0.002348720794543624
re -0.011364794336259365
sst2 -0.01410999707877636
rte -0.015595166012644768

EuclideanDistances:
WNLI 8.82930850982666
STSB 9.998733520507812
QNLI 20.563934326171875
rte 21.591501235961914
sst2 28.257226943969727
QQP 38.266563415527344
MNLI 39.47811508178711
re 51.698219299316406



---
QNLI
CosineSimilarity:
QQP 0.02450433000922203
MNLI 0.022226665169000626
rte 0.022116217762231827
STSB 0.019564714282751083
WNLI 0.018548090010881424
sst2 0.01816180720925331
MRPC 0.016551252454519272
re 0.005663424264639616

EuclideanDistances:
WNLI 20.32745933532715
MRPC 20.563934326171875
STSB 20.789505004882812
rte 27.955751419067383
sst2 33.33351516723633
QQP 42.09246826171875
MNLI 43.1364860534668
re 54.68842315673828



---
QQP
CosineSimilarity:
MNLI 0.02906796522438526
QNLI 0.02450433000922203
sst2 0.02038535475730896
re 0.014833556488156319
rte 0.014439152553677559
MRPC 0.012579187750816345
STSB 0.00864148698747158
WNLI -0.007635802496224642

EuclideanDistances:
MRPC 38.266563415527344
WNLI 38.267784118652344
STSB 38.425575256347656
QNLI 42.09246826171875
rte 42.66611862182617
sst2 46.172767639160156
MNLI 53.41044235229492
re 63.15311050415039



---
WNLI
CosineSimilarity:
MRPC 0.028840098530054092
STSB 0.02511468715965748
QNLI 0.018548090010881424
MNLI 0.003950587939471006
sst2 0.0019883119966834784
rte -0.0022752464283257723
re -0.002854109276086092
QQP -0.007635802496224642

EuclideanDistances:
MRPC 8.82930850982666
STSB 9.41944694519043
QNLI 20.32745933532715
rte 21.278451919555664
sst2 27.981159210205078
QQP 38.267784118652344
MNLI 39.31630325317383
re 51.545379638671875



---
STSB
CosineSimilarity:
WNLI 0.02511468715965748
QNLI 0.019564714282751083
QQP 0.00864148698747158
MRPC 0.006784915458410978
MNLI 0.006048301700502634
sst2 -0.003573349444195628
re -0.011063598096370697
rte -0.01770864985883236

EuclideanDistances:
WNLI 9.41944694519043
MRPC 9.998733520507812
QNLI 20.789505004882812
rte 21.863563537597656
sst2 28.38043975830078
QQP 38.425575256347656
MNLI 39.55362319946289
re 51.80781173706055


