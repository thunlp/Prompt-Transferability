import os
import copy
import json
import argparse


def get_max_source_length(d):
    if d in ['sst2', 'imdb', 'rotten_tomatoes', 'mrpc']:
        return 128
    if d in ['qqp', 'mnli', 'qnli', 'rte', 'snli']:
        return 256
    
    return 128

model_mapping = {
    'bert-tiny': 'prajjwal1/bert-tiny',
    'bert-small': 'prajjwal1/bert-small',
    'bert-base': 'bert-base-uncased',
    'bert-large': 'bert-large-uncased',
    'roberta-base': 'roberta-base',
    'roberta-large': 'roberta-large'
}

seeds = [42, 87, 21]
datasets = ["mrpc", "cola", "sst2", "qnli", "rte",  "mnli", "qqp", "snli"]
models = ['bert-tiny', 'bert-small', 'bert-base', 'bert-large', 'roberta-base', 'roberta-large']


template = {
    "do_train": True,
    "do_eval": True,
    "do_test": True,
    "prompt_len": 100,
    "num_proj_layers": 1,
    "flatten_proj": True, 
    "max_steps": 500000,
    "eval_steps": 1000,
    "save_steps": 1000,
    "logging_steps": 200,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 128,
    "learning_rate": 5e-4,
    "lr_scheduler_type": 'constant',
    "warmup_steps": 0,
    
    "save_total_limit": 1,
    "predict_with_generate": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "combined_score",
    "greater_is_better": True,
    "evaluation_strategy": "steps",
    "overwrite_output_dir": True,
    "save_strategy": "steps",
}



for m in models:
    for d in datasets:
        for s in seeds:
    
            config = copy.deepcopy(template)
            out_dir = f'configs/{m}'
            os.makedirs(out_dir, exist_ok=True)
            
            config['seed'] = s
            config['output_dir'] = f'outputs/{m}/{d}_{s}'
            config['backbone'] = model_mapping[m]
            config['dataset'] = d

            # for i in ["job_name", "task_name", "eval_dataset_name", "test_dataset_name"]:
            #     config[i] = d
            config['max_source_length'] = get_max_source_length(d)

            with open(f'{out_dir}/{d}_{s}.json', 'w') as f:
                json.dump(config, f, indent=4)
