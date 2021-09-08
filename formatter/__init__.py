import logging

from .Basic import BasicFormatter
from .RTEFormatter import RTEFormatter
from .RTEPromptFormatter import RTEPromptFormatter
from .RTEPromptRobertaFormatter import RTEPromptRobertaFormatter
from .SST2PromptFormatter import SST2PromptFormatter
from .SST2PromptRobertaFormatter import SST2PromptRobertaFormatter
from .WikiREFormatter import WikiREFormatter
from .WikiREPromptFormatter import WikiREPromptFormatter
from .SQuADPromptRobertaFormatter import SQuADPromptRobertaFormatter
from .CoLAPromptRobertaFormatter import CoLAPromptRobertaFormatter
from .MRPCPromptRobertaFormatter import MRPCPromptRobertaFormatter
from .QQPPromptRobertaFormatter import QQPPromptRobertaFormatter
from .MNLIPromptRobertaFormatter import MNLIPromptRobertaFormatter
from .QNLIPromptRobertaFormatter import QNLIPromptRobertaFormatter
from .WNLIPromptRobertaFormatter import WNLIPromptRobertaFormatter
from .STSBPromptRobertaFormatter import STSBPromptRobertaFormatter
from .laptopPromptRobertaFormatter import laptopPromptRobertaFormatter
from .restaurantPromptRobertaFormatter import restaurantPromptRobertaFormatter
from .IMDBPromptRobertaFormatter import IMDBPromptRobertaFormatter
from .projectorPromptRobertaFormatter import projectorPromptRobertaFormatter
from .mutiGPU_STSBPromptRobertaFormatter import mutiGPU_STSBPromptRobertaFormatter
from .crossPromptFormatter import crossPromptFormatter
from .mlmPromptFormatter import mlmPromptFormatter
from .cross_mlmPromptFormatter import cross_mlmPromptFormatter
from .snliPromptRobertaFormatter import snliPromptRobertaFormatter
from .anliPromptRobertaFormatter import anliPromptRobertaFormatter
from .recastfactualityPromptRobertaFormatter import recastfactualityPromptRobertaFormatter
from .tweetevalsentimentPromptRobertaFormatter import tweetevalsentimentPromptRobertaFormatter
from .movierationalesPromptRobertaFormatter import movierationalesPromptRobertaFormatter
from .emobankarousalPromptRobertaFormatter import emobankarousalPromptRobertaFormatter
from .persuasivenessrelevancePromptRobertaFormatter import persuasivenessrelevancePromptRobertaFormatter
from .persuasivenessspecificityPromptRobertaFormatter import persuasivenessspecificityPromptRobertaFormatter
from .emobankdominancePromptRobertaFormatter import emobankdominancePromptRobertaFormatter
from .squinkyimplicaturePromptRobertaFormatter import squinkyimplicaturePromptRobertaFormatter
from .squinkyformalityPromptRobertaFormatter import squinkyformalityPromptRobertaFormatter
from .activate_neuronPromptRobertaFormatter import activate_neuronPromptRobertaFormatter



logger = logging.getLogger(__name__)


formatter_list = {
    "Basic": BasicFormatter,
    "RTE": RTEFormatter,
    "RTEPrompt": RTEPromptFormatter,
    "RTEPromptRoberta": RTEPromptRobertaFormatter,
    "SST2Prompt": SST2PromptFormatter,
    "SST2_PromptRoberta": SST2PromptRobertaFormatter,
    "SQuADPromptRoberta": SQuADPromptRobertaFormatter,
    "RE": WikiREFormatter,
    "REPrompt": WikiREPromptFormatter,
    "WikiREPromptRoberta": WikiREPromptFormatter,
    "CoLAPromptRoberta": CoLAPromptRobertaFormatter,
    "MRPCPromptRoberta": MRPCPromptRobertaFormatter,
    "QQPPromptRoberta": QQPPromptRobertaFormatter,
    "MNLIPromptRoberta": MNLIPromptRobertaFormatter,
    "QNLIPromptRoberta": QNLIPromptRobertaFormatter,
    "WNLIPromptRoberta": WNLIPromptRobertaFormatter,
    "STSBPromptRoberta": STSBPromptRobertaFormatter,
    "laptopPromptRoberta": laptopPromptRobertaFormatter,
    "restaurantPromptRoberta": restaurantPromptRobertaFormatter,
    "IMDBPromptRoberta": IMDBPromptRobertaFormatter,
    "projectorPromptRoberta": projectorPromptRobertaFormatter,
    "mutiGPU_STSBPromptRoberta": mutiGPU_STSBPromptRobertaFormatter,
    "crossPrompt": crossPromptFormatter,
    "mlmPrompt": mlmPromptFormatter,
    "cross_mlmPrompt": cross_mlmPromptFormatter,
    "snliPromptRoberta": snliPromptRobertaFormatter,
    "anliPromptRoberta": snliPromptRobertaFormatter,
    "recastfactualityPromptRoberta": recastfactualityPromptRobertaFormatter,
    "tweetevalsentimentPromptRoberta": tweetevalsentimentPromptRobertaFormatter,
    "movierationalesPromptRoberta": movierationalesPromptRobertaFormatter,
    "emobankarousalPromptRoberta": emobankarousalPromptRobertaFormatter,
    "persuasivenessrelevancePromptRoberta": persuasivenessrelevancePromptRobertaFormatter,
    "persuasivenessspecificityPromptRoberta": persuasivenessspecificityPromptRobertaFormatter,
    "emobankdominancePromptRoberta": emobankdominancePromptRobertaFormatter,
    "squinkyimplicaturePromptRoberta": squinkyimplicaturePromptRobertaFormatter,
    "squinkyformalityPromptRoberta": squinkyformalityPromptRobertaFormatter,
    "activate_neuronPromptRoberta": activate_neuronPromptRobertaFormatter,
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)



    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)
        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
