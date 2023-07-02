import os
import sys
import torch
import logging
import random
import numpy as np

from datasets import load_dataset, load_metric
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    set_seed,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
from openprompt.data_utils.utils import InputExample
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, ManualVerbalizer

from prompt_hub import task_to_keys, get_model
from prompt_hub.hub import PromptHub
from prompt_hub.training_args import PromptTrainingArguments, RemainArgHfArgumentParser


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = RemainArgHfArgumentParser((PromptTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        json_file=os.path.abspath(sys.argv[1])
        args, _ = parser.parse_json_file(json_file, return_remaining_args=True) #args = arg_string, return_remaining_strings=True) #parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)

    # Dataset
    is_regression = args.dataset in ['stsb']
    # raw_dataset = load_dataset("glue", self.args.dataset)
    # train_dataset = [InputExample(guid=e['idx'], text_a=e['question'], text_b=e['sentence'], label=e['label']) for e in raw_dataset['train']]#[:100]
    # eval_dataset = [InputExample(guid=e['idx'], text_a=e['question'], text_b=e['sentence'], label=e['label']) for e in raw_dataset['validation']]#[:100]

    # Model
    # plm, tokenizer, model_config, tokenizer_wrapper_class = load_plm('roberta', args.backbone)
    # template = '{"soft": None, "duplicate": ' + str(args.prompt_len) + ', "same": True} {"mask"} {"placeholder": "text_a"} {"placeholder": "text_b"}'
    # template = SoftTemplate(model=plm, text=template, tokenizer=tokenizer, num_tokens=args.prompt_len) # initialize_from_vocab=args.init_from_vocab
    # verbalizer = ManualVerbalizer(tokenizer, classes=raw_dataset['train'].features['label'].names).from_file(f'verbalizer/{args.dataset}.txt', choice=0)
    # model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=True)

    metric = load_metric("prompt_hub/glue_metrics.py", args.dataset)


    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        result["combined_score"] = np.mean(list(result.values())).item()

        return result



    # Train
    trainer = PromptHub(
        args=args,
        compute_metrics=compute_metrics,
    )



    train_results = trainer.train_prompt(args.backbone, args.dataset)
    print(train_results)

    eval_results = trainer.eval_prompt(args.backbone, args.dataset)
    print(eval_results)

    cross_task_results = trainer.cross_task_eval(args.backbone, 'rotten_tomatoes')
    print(cross_task_results)

    trainer.cross_model_train(args.backbone, 'roberta-large', args.dataset)
    trainer.cross_task_eval(args.backbone, 'roberta-large', args.dataset)

    # Trainer
    # data_collator = DataCollatorWithPadding(tokenizer, max_length=args.max_source_length, pad_to_multiple_of=8)
    # model=model,
    # template=template,
    # verbalizer=verbalizer,
    # tokenizer_wrapper_class=tokenizer_wrapper_class,
    # train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    # tokenizer=tokenizer,
    # classes=raw_dataset['train'].features['label'].names
    # data_collator=data_collator

    # if args.do_train:
    #     # Detecting last checkpoint.
    #     last_checkpoint = None
    #     if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #         last_checkpoint = get_last_checkpoint(args.output_dir)
    #         if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
    #             raise ValueError(
    #                 f"Output directory ({args.output_dir}) already exists and is not empty. "
    #                 "Use --overwrite_output_dir to overcome."
    #             )
    #         elif last_checkpoint is not None and args.resume_from_checkpoint is None:
    #             logger.info(
    #                 f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #                 "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #             )
    #     train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    #     metrics = train_result.metrics
    #     # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()

    # results = {}
    # # Evaluation
    # if args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         eval_datasets.append(raw_datasets["validation_mismatched"])

    #     for eval_dataset, task in zip(eval_datasets, tasks):
    #         metrics = trainer.evaluate(eval_dataset=eval_dataset)

    #         max_eval_samples = (
    #             data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #         )
    #         metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #         trainer.log_metrics("eval", metrics)
    #         trainer.save_metrics("eval", metrics)
    #     results['eval'] = metrics

    # if args.do_predict:
    #     logger.info("*** Predict ***")

    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     tasks = [data_args.task_name]
    #     predict_datasets = [predict_dataset]
    #     if data_args.task_name == "mnli":
    #         tasks.append("mnli-mm")
    #         predict_datasets.append(raw_datasets["test_mismatched"])

    #     for predict_dataset, task in zip(predict_datasets, tasks):
    #         # Removing the `label` columns because it contains -1 and Trainer won't like that.
    #         predict_dataset = predict_dataset.remove_columns("label")
    #         predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    #         predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

    #         output_predict_file = os.path.join(args.output_dir, f"predict_results_{task}.txt")
    #         if trainer.is_world_process_zero():
    #             with open(output_predict_file, "w") as writer:
    #                 logger.info(f"***** Predict results {task} *****")
    #                 writer.write("index\tprediction\n")
    #                 for index, item in enumerate(predictions):
    #                     if is_regression:
    #                         writer.write(f"{index}\t{item:3.3f}\n")
    #                     else:
    #                         item = label_list[item]
    #                         writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()
