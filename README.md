# On Transferability of Prompt Tuning for Natural Language Processing

[![Version](https://img.shields.io/badge/Version-v0.1.0-blue?color=FF8000?color=009922)](https://img.shields.io/badge/Version-v0.1.0-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2111.06719-B21A1B)](https://arxiv.org/abs/2111.06719)
[![GitHub Stars](https://img.shields.io/github/stars/thunlp/Prompt-Transferability?style=social)](https://github.com/thunlp/Prompt-Transferability/stargazers)


[**NAACL 2022**](https://2022.naacl.org/) Accepted. (link: https://arxiv.org/abs/2111.06719)



## Overview
![prompt_transferability](github_fig/prompt_transferbility_github.png)

Prompt tuning (PT) is a promising parameter-efficient method to utilize extremely large pre-trained language models (PLMs), which can achieve comparable performance to full-parameter fine-tuning by only tuning a few soft prompts. However, PT requires much more training time than fine-tuning. Intuitively, knowledge transfer can help to improve the efficiency. To explore whether we can improve PT via prompt transfer, we empirically investigate the transferability of soft prompts across different downstream tasks and PLMs in this work. We find that (1) in zero-shot setting, trained soft prompts can effectively transfer to similar tasks on the same PLM and also to other PLMs with a cross-model projector trained on similar tasks; (2) when used as initialization, trained soft prompts of similar tasks and projected prompts of other PLMs can significantly accelerate training and also improve the performance of PT. Moreover, to explore what decides prompt transferability, we investigate various transferability indicators and find that the overlapping rate of activated neurons strongly reflects the transferability, which suggests how the prompts stimulate PLMs is essential. Our findings show that prompt transfer is promising for improving PT, and further research shall focus more on prompts' stimulation to PLMs.

### Setups
* pip>=21.3.1
* python>=3.6.13
* torch==1.9.0+cu111

You could refer `environment.yml` for more details.


### Requirements
```
bash requirement.sh
```

### Download PLM Checkpoints
```
```

### Download Downstream Dataset
```
```

## Train (Prompt Tuning)
Perform Prompt Tuning: 
```
bash train.sh
```

Example:
```
gpus=0
DATASET="SST2"
BACKBONE="Roberta"

CUDA_VISIBLE_DEVICES=$gpus python3 train.py --config config/${DATASET}Prompt${BACKBONE}.config \
    --gpu $gpus \
    --checkpoint model/${DATASET}Prompt${BACKBONE} \
```

Main Arguments:
* `--config`: Prompt Tuning (PT) / Validation configurations in `/config` directory.
* `--gpu`: Assign gpu ID.
* `--checkpoint`: Initialize prompt parameters for Prompt Tuning (PT).


## Evaluate (Trained Prompts)
Evaluate trained prompts on the corresponding tasks:
```
bash valid.sh
```

Example:
```
gpus=0
DATASET="SST2"
BACKBONE="Roberta"

CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/{DATASET}Prompt${BACKBONE}.config \
    --gpu $gpus \
    --checkpoint model/${DATASET}Prompt${BACKBONE} \
```
Main Arguments:
* `--config`: Utilize the same configurations as PT.
* `--gpu`: Assign gpu ID.
* `--checkpoint`: Trained prompts for validation.



## Cross-task Transfer
Perform cross-task transfer experiments: 
```
bash valid_cross_task.sh
```

Example:
```
gpus=0
BASEMODEL="T5Large"
DATASET=IMDBPrompt 
PROMPT=laptopPrompt 

echo "==========================="
echo Model: config/${DATASET}${BASEMODEL}.config
echo Prompt-emb: task_prompt_emb/${PROMPT}${BASEMODEL}
echo "==========================="

CUDA_VISIBLE_DEVICES=$gpus python3 valid.py --config config/${DATASET}${BASEMODEL}.config \
    --gpu $gpus \
    --checkpoint model/${DATASET}${BASEMODEL} \
    --replacing_prompt task_prompt_emb/${PROMPT}${BASEMODEL}
```

Main Arguments:
* `--config`: Utilize the same configurations as PT.
* `--gpu`: Assign gpu ID.
* `--checkpoint`: Trained prompts for validation.
