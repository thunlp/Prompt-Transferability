import logging

from .Basic import BasicFormatter
from .RTEFormatter import RTEFormatter
from .RTEPromptFormatter import RTEPromptFormatter
from .RTEPromptRobertaFormatter import RTEPromptRobertaFormatter
from .SST2PromptFormatter import SST2PromptFormatter
from .SST2PromptRobertaFormatter import SST2PromptRobertaFormatter
from .SST2PromptT5Formatter import SST2PromptT5Formatter
from .WikiREFormatter import WikiREFormatter
from .WikiREPromptFormatter import WikiREPromptFormatter
from .SQuADPromptRobertaFormatter import SQuADPromptRobertaFormatter
from .CoLAPromptRobertaFormatter import CoLAPromptRobertaFormatter
from .MRPCPromptRobertaFormatter import MRPCPromptRobertaFormatter
from .MRPCPromptT5Formatter import MRPCPromptT5Formatter
from .QQPPromptRobertaFormatter import QQPPromptRobertaFormatter
from .QQPPromptT5Formatter import QQPPromptT5Formatter
from .MNLIPromptRobertaFormatter import MNLIPromptRobertaFormatter
from .MNLIPromptT5Formatter import MNLIPromptT5Formatter
from .QNLIPromptRobertaFormatter import QNLIPromptRobertaFormatter
from .QNLIPromptT5Formatter import QNLIPromptT5Formatter
from .WNLIPromptRobertaFormatter import WNLIPromptRobertaFormatter
from .STSBPromptRobertaFormatter import STSBPromptRobertaFormatter
from .laptopPromptRobertaFormatter import laptopPromptRobertaFormatter
from .laptopPromptT5Formatter import laptopPromptT5Formatter
from .restaurantPromptRobertaFormatter import restaurantPromptRobertaFormatter
from .restaurantPromptT5Formatter import restaurantPromptT5Formatter
from .IMDBPromptRobertaFormatter import IMDBPromptRobertaFormatter
from .IMDBPromptT5Formatter import IMDBPromptT5Formatter
from .projectorPromptRobertaFormatter import projectorPromptRobertaFormatter
from .mutiGPU_STSBPromptRobertaFormatter import mutiGPU_STSBPromptRobertaFormatter
from .crossPromptFormatter import crossPromptFormatter
from .mlmPromptFormatter import mlmPromptFormatter
from .cross_mlmPromptFormatter import cross_mlmPromptFormatter
from .snliPromptRobertaFormatter import snliPromptRobertaFormatter
from .snliPromptT5Formatter import snliPromptT5Formatter
from .anliPromptRobertaFormatter import anliPromptRobertaFormatter
from .recastfactualityPromptRobertaFormatter import recastfactualityPromptRobertaFormatter
from .recastpunsPromptRobertaFormatter import recastpunsPromptRobertaFormatter
from .recastverbnetPromptRobertaFormatter import recastverbnetPromptRobertaFormatter
from .recastverbcornerPromptRobertaFormatter import recastverbcornerPromptRobertaFormatter
from .recastnerPromptRobertaFormatter import recastnerPromptRobertaFormatter
from .recastsentimentPromptRobertaFormatter import recastsentimentPromptRobertaFormatter
from .recastmegaveridicalityPromptRobertaFormatter import recastmegaveridicalityPromptRobertaFormatter
from .tweetevalsentimentPromptRobertaFormatter import tweetevalsentimentPromptRobertaFormatter
from .tweetevalsentimentPromptT5Formatter import tweetevalsentimentPromptT5Formatter
from .movierationalesPromptRobertaFormatter import movierationalesPromptRobertaFormatter
from .movierationalesPromptT5Formatter import movierationalesPromptT5Formatter
from .emobankarousalPromptRobertaFormatter import emobankarousalPromptRobertaFormatter
from .persuasivenessrelevancePromptRobertaFormatter import persuasivenessrelevancePromptRobertaFormatter
from .persuasivenessspecificityPromptRobertaFormatter import persuasivenessspecificityPromptRobertaFormatter
from .emobankdominancePromptRobertaFormatter import emobankdominancePromptRobertaFormatter
from .squinkyimplicaturePromptRobertaFormatter import squinkyimplicaturePromptRobertaFormatter
from .squinkyformalityPromptRobertaFormatter import squinkyformalityPromptRobertaFormatter
from .activate_neuronPromptRobertaFormatter import activate_neuronPromptRobertaFormatter
from .ethicscommonsensePromptRobertaFormatter import ethicscommonsensePromptRobertaFormatter
from .ethicsdeontologyPromptRobertaFormatter import ethicsdeontologyPromptRobertaFormatter
from .ethicsdeontologyPromptT5Formatter import ethicsdeontologyPromptT5Formatter
from .ethicsjusticePromptRobertaFormatter import ethicsjusticePromptRobertaFormatter
from .ethicsjusticePromptT5Formatter import ethicsjusticePromptT5Formatter
from .ethicsvirtuePromptRobertaFormatter import ethicsvirtuePromptRobertaFormatter
from .squadPromptT5Formatter import squadPromptT5Formatter
from .nq_openPromptT5Formatter import nq_openPromptT5Formatter
from .multi_newsPromptT5Formatter import multi_newsPromptT5Formatter
from .samsumPromptT5Formatter import samsumPromptT5Formatter



logger = logging.getLogger(__name__)


formatter_list = {
    "Basic": BasicFormatter,
    "RTE": RTEFormatter,
    "RTEPrompt": RTEPromptFormatter,
    "RTEPromptRoberta": RTEPromptRobertaFormatter,
    "SST2Prompt": SST2PromptFormatter,
    "SST2_PromptRoberta": SST2PromptRobertaFormatter,
    "SST2_PromptT5": SST2PromptT5Formatter,
    "SQuADPromptRoberta": SQuADPromptRobertaFormatter,
    "RE": WikiREFormatter,
    "REPrompt": WikiREPromptFormatter,
    "WikiREPromptRoberta": WikiREPromptFormatter,
    "CoLAPromptRoberta": CoLAPromptRobertaFormatter,
    "MRPCPromptRoberta": MRPCPromptRobertaFormatter,
    "MRPCPromptT5": MRPCPromptT5Formatter,
    "QQPPromptRoberta": QQPPromptRobertaFormatter,
    "QQPPromptT5": QQPPromptT5Formatter,
    "MNLIPromptRoberta": MNLIPromptRobertaFormatter,
    "MNLIPromptT5": MNLIPromptT5Formatter,
    "QNLIPromptRoberta": QNLIPromptRobertaFormatter,
    "QNLIPromptT5": QNLIPromptT5Formatter,
    "WNLIPromptRoberta": WNLIPromptRobertaFormatter,
    "STSBPromptRoberta": STSBPromptRobertaFormatter,
    "laptopPromptRoberta": laptopPromptRobertaFormatter,
    "laptopPromptT5": laptopPromptT5Formatter,
    "restaurantPromptRoberta": restaurantPromptRobertaFormatter,
    "restaurantPromptT5": restaurantPromptT5Formatter,
    "IMDBPromptRoberta": IMDBPromptRobertaFormatter,
    "IMDBPromptT5": IMDBPromptT5Formatter,
    "projectorPromptRoberta": projectorPromptRobertaFormatter,
    "mutiGPU_STSBPromptRoberta": mutiGPU_STSBPromptRobertaFormatter,
    "crossPrompt": crossPromptFormatter,
    "mlmPrompt": mlmPromptFormatter,
    "cross_mlmPrompt": cross_mlmPromptFormatter,
    "snliPromptRoberta": snliPromptRobertaFormatter,
    "snliPromptT5": snliPromptT5Formatter,
    "anliPromptRoberta": anliPromptRobertaFormatter,
    "recastfactualityPromptRoberta": recastfactualityPromptRobertaFormatter,
    "recastpunsPromptRoberta": recastpunsPromptRobertaFormatter,
    "recastverbnetPromptRoberta": recastverbnetPromptRobertaFormatter,
    "recastverbcornerPromptRoberta": recastverbcornerPromptRobertaFormatter,
    "recastnerPromptRoberta": recastnerPromptRobertaFormatter,
    "recastsentimentPromptRoberta": recastsentimentPromptRobertaFormatter,
    "recastmegaveridicalityPromptRoberta": recastmegaveridicalityPromptRobertaFormatter,
    "tweetevalsentimentPromptRoberta": tweetevalsentimentPromptRobertaFormatter,
    "tweetevalsentimentPromptT5": tweetevalsentimentPromptT5Formatter,
    "movierationalesPromptRoberta": movierationalesPromptRobertaFormatter,
    "movierationalesPromptT5": movierationalesPromptT5Formatter,
    "emobankarousalPromptRoberta": emobankarousalPromptRobertaFormatter,
    "persuasivenessrelevancePromptRoberta": persuasivenessrelevancePromptRobertaFormatter,
    "persuasivenessspecificityPromptRoberta": persuasivenessspecificityPromptRobertaFormatter,
    "emobankdominancePromptRoberta": emobankdominancePromptRobertaFormatter,
    "squinkyimplicaturePromptRoberta": squinkyimplicaturePromptRobertaFormatter,
    "squinkyformalityPromptRoberta": squinkyformalityPromptRobertaFormatter,
    "activate_neuronPromptRoberta": activate_neuronPromptRobertaFormatter,
    "ethicscommonsensePromptRoberta": ethicscommonsensePromptRobertaFormatter,
    "ethicsdeontologyPromptRoberta": ethicsdeontologyPromptRobertaFormatter,
    "ethicsdeontologyPromptT5": ethicsdeontologyPromptT5Formatter,
    "ethicsjusticePromptRoberta": ethicsjusticePromptRobertaFormatter,
    "ethicsjusticePromptT5": ethicsjusticePromptT5Formatter,
    "ethicsvirtuePromptRoberta": ethicsvirtuePromptRobertaFormatter,
    "squadPromptT5": squadPromptT5Formatter,
    "nq_openPromptT5": nq_openPromptT5Formatter,
    "multi_newsPromptT5": multi_newsPromptT5Formatter,
    "samsumPromptT5": samsumPromptT5Formatter,
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
