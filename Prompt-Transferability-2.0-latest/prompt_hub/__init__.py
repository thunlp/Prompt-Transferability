from .models.bert import PromptBert
from .models.roberta import PromptRoberta


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "amazon_polarity": ("text", None),
    "rotten_tomatoes": ("text", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}



def get_model(args):
    if 'bert-' in args.backbone:
        return PromptBert(args)

    if 'roberta-' in args.backbone:
        return PromptRoberta(args)
