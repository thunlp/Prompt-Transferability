
# On Transferability of Prompt Tuning for Natural Language Processing (Prompt Transferability)
[![Version](https://img.shields.io/badge/Version-v0.1.0-blue?color=FF8000?color=009922)](https://img.shields.io/badge/Version-v0.1.0-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.18653/v1/2022.naacl-green?color=FF8000?color=009922)](https://aclanthology.org/2022.naacl-main.290)
[![GitHub Stars](https://img.shields.io/github/stars/thunlp/Prompt-Transferability?style=social)](https://github.com/thunlp/Prompt-Transferability/stargazers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xUe9rLc2K9EbFAX9iDO1x9j9ZRKoUeO-?usp=sharing)
<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VCSIDaX_pgkrSjzouaNH14D8Fo7G9GBz?usp=sharing)-->


This is the source code of "On Transferability of Prompt Tuning for Natural Language Processing", an [NAACL 2022](https://2022.naacl.org/) paper [[**pdf**]](https://aclanthology.org/2022.naacl-main.290/).

## Overview
![prompt_transferability](github_profile/prompt_transferbility_github.png)


Prompt tuning (PT) is a promising parameter-efficient method to utilize extremely large pre-trained language models (PLMs), which can achieve comparable performance to full-parameter fine-tuning by only tuning a few soft prompts. However, PT requires much more training time than fine-tuning. Intuitively, knowledge transfer can help to improve the efficiency. To explore whether we can improve PT via prompt transfer, we empirically investigate the transferability of soft prompts across different downstream tasks and PLMs in this work. We find that (1) in zero-shot setting, trained soft prompts can effectively transfer to similar tasks on the same PLM and also to other PLMs with a cross-model projector trained on similar tasks; (2) when used as initialization, trained soft prompts of similar tasks and projected prompts of other PLMs can significantly accelerate training and also improve the performance of PT. Moreover, to explore what decides prompt transferability, we investigate various transferability indicators and find that the overlapping rate of activated neurons strongly reflects the transferability, which suggests how the prompts stimulate PLMs is essential. Our findings show that prompt transfer is promising for improving PT, and further research shall focus more on prompts' stimulation to PLMs.


## Reproduce results in the paper

- [Prompt-Transferability-1.0](./Prompt-Transferability-1.0/) provides the original codes and details to reproduce the results in the paper.
- [Prompt-Transferability-2.0-latest](https://github.com/thunlp/Prompt-Transferability/tree/main) refactors the Prompt-Transferability-1.0 and provides more user-friendly codes for users. In this `README.md`, we mainly demostrate the usage of the `Prompt-Transferability-2.0-latest`.  



## Setups
- python==3.8.0

We recommend to create a new Anaconda environment to manage the required packages. 
```bash
conda create -n prompt_transfer python=3.8.0
conda activate prompt_transfer
pip install -r requirements.txt
```
If the system shows `ERROR: Invalid requirement: 'torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0' (from line 10 of requirements.txt)`. Please manually run `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`.


User can also directly create the environment via `Prompt-Transferability-2.0-latest/environment.yml`.
```bash
conda env create -f environment.yml
```



## Usage

You can easily use PromptHub for various perposes, including prompt training, evaluation, cross-task transfer, cross-model transfer, and activated neuron. The [Colab notebook](https://colab.research.google.com/drive/1xUe9rLc2K9EbFAX9iDO1x9j9ZRKoUeO-?usp=sharing) and the [example script](./Prompt-Transferability-2.0-latest/example/train.py) also demonstrate the usages. Or, you can run the bash file to run a quick example.
```bash
cd Prompt-Transferability-2.0-latest
bash example/train.sh
```

The above code shows an example of prompt training (based on `Roberta-base`), evaluation, activated neuron analysis on `SST2` dataset.

```python
from prompt_hub.hub import PromptHub
from prompt_hub.training_args import PromptTrainingArguments

# Training config
args = PromptTrainingArguments(
  output_dir='outputs', 
  dataset='sst2', 
  backbone='roberta-base', 
  learning_rate=1e-2
)
trainer = PromptHub(args=args)

# Prompt training and evaluation
trainer.train_prompt()
trainer.eval_prompt()

# Cross-task evaluation
cross_task_eval_results = trainer.cross_task_eval('roberta-base', 'sst2', 'rotten_tomatoes')


# Activated neuron
activated_neuron_before_relu, activated_neuron_after_relu = trainer.activated_neuron(args.backbone, args.dataset)
```

Or, run the following command to start to run a quick example (from `Prompt-Transferability-2.0-latest/example/train.sh`)
```bash
python example/train.py \
        --output_dir outputs \
        --dataset sst2 \
        --learning_rate 1e-2 \
        --num_train_epochs 3 \
        --save_total_limit 1 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end true \
        --metric_for_best_model combined_score
```

## Dataset
Excute `Prompt-Transferability-2.0-latest/download_data.sh` to download datasets.
```bash
cd Prompt-Transferability-2.0-latest
bash download_data.sh
```

## Detailed Usage

### Prompt Tuning
![prompt_transferability](github_profile/prompt_tuning.png)

Users can use the well-trained prompts in `Prompt-Transferability-2.0-latest/task_prompt_emb` or re-train the prompts by your own.


#### Step 1: initialization of arguments and trainer
We first need to define a set of arguments or configurations, including `backbone` (backbone model), `dataset`, `prompt_len` (the length of soft prompt), etc. Then we instantiate a `PromptHub` object passing in the arguments we just created.

```python
from prompt_hub.training_args import PromptTrainingArguments

args = PromptTrainingArguments(
  output_dir='outputs',
  backbone='roberta-base',
  dataset='sst2',
  prompt_len=100
)
trainer = PromptHub(args=args)
```

For a complete list of arguments, please refer to `Prompt-Transferability-2.0-latest/prompt_hub/training_args.py` and HuggingFace `transformers.training_arguments` for more details. 

#### Step 2: prompt training
Then we start training a soft prompt. (_Optional_)You can pass in parameters to overwrite the default configurations in the arguments you passed in. This framework supports all datasets and some backbone models (`Bert`, `Roberta`, `GPT`, and `T5 v1.1`), currently. 

```python
# Optional arguments to overwrite default parameters
# trainer.train_prompt('roberta-large', 'sst2') 
trainer.train_prompt() 
```

#### Step 3: prompt evaluation
With the prompt (trained on specific dataset and utilized backbone model), we excute the following code to evaluate its performance. 

```python
# Optional arguments to overwrite default parameters
# eval_results = trainer.eval_prompt('roberta-base', 'sst2') 
eval_results = trainer.eval_prompt()
```


### Cross-Task Transfer
![prompt_transferability](github_profile/cross_task.gif)
Prompt can directly transfer among tasks. Here, we provide an example to transfer the prompt trained from `SST2` dataset to `Rotten Tomatoes` dataset.

```python
cross_task_eval_results = trainer.cross_task_eval('roberta-base', 'sst2', 'rotten_tomatoes')
```

### Cross-Model Transfer
![prompt_transferability](github_profile/cross_model.gif)
Prompt can utilize a well-trained projector to transfer among different backbones. 

#### Step 1: cross-model Training
We first train a projector (from `roberta-base` to `roberta-large` on `SST2` dataset).

```python
trainer.cross_model_train(source_model='roberta-base', target_model='roberta-large', task='sst2')
```

#### Step 2: cross-model evaluation
Then, we utilize it to transfer the prompt among different models. 

```python
cross_model_eval_results = trainer.cross_model_eval(source_model='roberta-base', target_model='roberta-large', task='sst2')
```


### Transferability Indicators (Activated neuron)
![prompt_transferability](github_profile/activated_neurons.gif)
Prompt can be seen as a paradigm to manipulate PLMs (stimulate artificial neurons) knowledge to perform downstream tasks. We further observe that similar prompts will activate similar neurons; thus, the activated neurons can be a transferability indicator.

Definition of Neurons: the output values between 1st and 2nd layers of feed-forward network FFN (in every layer of a PLM) [Refer to Section 6.1 in the paper]

#### Step 1: Acquire task-specific neurons
Given a model and the trained task-specific prompt, you can obtain the activated neurons values.

```python
activated_neuron_before_relu, activated_neuron_after_relu = trainer.activated_neuron('roberta-base', 'sst2')
```

#### Step 2: Similarity/Transferability between two tasks
You can caculate the similarity/transferability between two prompts via actiaved neurons.
```python
cos_sim = trainer.neuron_similarity(backbone='roberta-base', task1='sst2', task2='rotten_tomatoes')
```

#### Step 3: Masked Neurons
To further demonstrate the importance of task-specific neurons, we mask them and find the model performance on the corresponding task will degrade. Visualization of activated neurons is also supported.

```python
eval_metric, mask = trainer.mask_activated_neuron(args.backbone, args.dataset, ratio=0.2)
trainer.plot_neuron()
```

<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VCSIDaX_pgkrSjzouaNH14D8Fo7G9GBz?usp=sharing)-->

## Citations
[![DOI](https://img.shields.io/badge/DOI-10.18653/v1/2022.naacl-green?color=FF8000?color=009922)](https://aclanthology.org/2022.naacl-main.290)

Please cite our paper if it is helpful to your work!

```bibtex
@inproceedings{su-etal-2022-transferability,
    title = "On Transferability of Prompt Tuning for Natural Language Processing",
    author = "Su, Yusheng  and
      Wang, Xiaozhi  and
      Qin, Yujia  and
      Chan, Chi-Min  and
      Lin, Yankai  and
      Wang, Huadong  and
      Wen, Kaiyue  and
      Liu, Zhiyuan  and
      Li, Peng  and
      Li, Juanzi  and
      Hou, Lei  and
      Sun, Maosong  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.290",
    doi = "10.18653/v1/2022.naacl-main.290",
    pages = "3949--3969"
}
```

## Contact
[Yusheng Su](https://yushengsu-thu.github.io/)

Mail: yushengsu.thu@gmail.com; suys19@mauls.tsinghua.edu.cn
