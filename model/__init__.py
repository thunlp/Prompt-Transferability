from .BinaryClsBERT import BinaryClsBERT,PromptClsBERT
from .REBERT import REBERT
from .REPromptRoberta import REPromptRoberta
from .PromptRoberta import PromptRoberta
from .projectPromptRoberta import projectPromptRoberta
from .projectPromptRoberta_prompt import projectPromptRoberta_prompt
#from .SQuADPromptRoberta import SQuADPromptRoberta
from .PromptBert import PromptBert
from .crossPromptRoberta import crossPromptRoberta
from .mlmPrompt import mlmPrompt
model_list = {
    "BinaryClsBERT": BinaryClsBERT,
    "RTEPrompt": PromptClsBERT,
    "SST2Prompt": PromptClsBERT,
    "RE": REBERT,
    "REPrompt": REPromptRoberta,
    "projectPromptRoberta": projectPromptRoberta,
    "PromptRoberta": PromptRoberta,
    "projectPromptRoberta_prompt": projectPromptRoberta_prompt,
    "crossPromptRoberta": crossPromptRoberta,
    #"SQuADPromptRoberta": SQuADPromptRoberta
    "PromptBert": PromptBert,
    "mlmPrompt": mlmPrompt
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
