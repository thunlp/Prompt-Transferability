import logging
import torch

from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from .output_init import init_output_function
from torch import nn
from transformers import AutoTokenizer
#from tools.return_fun import return_conf

logger = logging.getLogger(__name__)

#def return_conf(config):
#    return config

def init_all(config, gpu_list, checkpoint, mode, *args, **params):

    result = {}

    '''
    for key in config.sections():
        for val in config.items(key):
            print(key, val)
    '''

    '''
    logger.info("Begin to initialize dataset and formatter...")
    if mode == "train":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)

    else:
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)
    '''

    logger.info("Begin to initialize models...")



    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    global_step = 0


    if len(gpu_list) > 0:
        if params['local_rank'] < 0:
            model = model.cuda()
        else:
            ###
            #muti machines
            #model = model.to(gpu_list[params['local_rank']])

            #single machine
            model = model.to(params['local_rank'])
            ###


        try:
            ###
            #muti machines
            model = nn.parallel.DistributedDataParallel(model, device_ids=[params['local_rank']], output_device=params['local_rank'], find_unused_parameters = True)

            #single machine
            #model = nn.parallel.DistributedDataParallel(model, device_ids=gpu_list)
            #model = nn.parallel.DistributedDataParallel(model)
            ###
        except Exception as e:
            logger.warning("No init_multi_gpu implemented in the model, use single gpu instead.")


    if config.getboolean("prompt", "prompt_tune") and config.get("model", "model_name") == "SQuADPromptRoberta":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        init_ids = [] #tokenizer.encode("the relation between the first sentence and the second sentence is")
        pad_num = config.getint("prompt", "prompt_num") - len(init_ids)
        init_ids.extend([tokenizer.mask_token_id] * pad_num)
        if hasattr(model, 'module'):
            model.module.init_prompt_emb(init_ids)
        else:
            model.init_prompt_emb(init_ids)

    try:
        parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters["model"])
        else:
            model.load_state_dict(parameters["model"])

        if mode == "train":
            trained_epoch = parameters["trained_epoch"]
            if config.get("train", "optimizer") == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logger.warning("Optimizer changed, do not load parameters of optimizer.")

            if "global_step" in parameters:
                global_step = parameters["global_step"]

    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)

    result["model"] = model
    if mode == "train":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    logger.info("Initialize done.")


    return result
