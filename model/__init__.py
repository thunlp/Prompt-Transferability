from .BinaryClsBERT import BinaryClsBERT,PromptClsBERT
from .REBERT import REBERT
from .REPromptRoberta import REPromptRoberta
from .PromptRoberta import PromptRoberta
from .projectPromptRoberta import projectPromptRoberta
#from .SQuADPromptRoberta import SQuADPromptRoberta
model_list = {
    "BinaryClsBERT": BinaryClsBERT,
    "RTEPrompt": PromptClsBERT,
    "SST2Prompt": PromptClsBERT,
    "RE": REBERT,
    "REPrompt": REPromptRoberta,
    "projectPromptRoberta": projectPromptRoberta,
    "PromptRoberta": PromptRoberta,
    #"SQuADPromptRoberta": SQuADPromptRoberta
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
