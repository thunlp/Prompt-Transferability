from .BinaryClsBERT import BinaryClsBERT,PromptClsBERT
from .PromptRoberta import PromptRoberta
from .REBERT import REBERT
from .REPromptRoberta import REPromptRoberta
#from .SQuADPromptRoberta import SQuADPromptRoberta
model_list = {
    "BinaryClsBERT": BinaryClsBERT,
    "RTEPrompt": PromptClsBERT,
    "SST2Prompt": PromptClsBERT,
    "PromptRoberta": PromptRoberta,
    "RE": REBERT,
    "REPrompt": REPromptRoberta,
    #"SQuADPromptRoberta": SQuADPromptRoberta
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
