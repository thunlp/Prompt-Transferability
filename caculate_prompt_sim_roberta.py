# -*- coding: utf-8 -*-
#import numpy as np
import torch
import config
#from activate_neuron.mymodel import *
#import activate_neuron.mymodel as mymodel
#from activate_neuron.utils import *
#import activate_neuron.utils as utils


#from transformers import AutoConfig, AutoModelForMaskedLM
#from model.modelling_roberta import RobertaForMaskedLM
#from reader.reader import init_dataset, init_formatter, init_test_dataset

import argparse
import os
import torch
import logging
import random
import numpy as np

from tools.init_tool import init_all
from config_parser import create_config
from tools.valid_tool import valid
from torch.autograd import Variable

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def relu(tmp):
    return 1*(tmp > 0)*tmp

def topk(obj, k):
    M=-10000
    obj = list(obj)[:]
    idlist = []
    for i in range(k):
        idlist.append(obj.index(max(obj)))
        obj[obj.index(max(obj))]=M
    return idlist

def relu(tmp):
    return 1*(tmp > 0)*tmp

def topk(obj, k):
    M=-10000
    obj = list(obj)[:]
    idlist = []
    for i in range(k):
        idlist.append(obj.index(max(obj)))
        obj[obj.index(max(obj))]=M
    return idlist




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--checkpoint', help="checkpoint file path", type=str, default=None)
    parser.add_argument('--comment', help="checkpoint file path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt_emb_output", type=bool, default=False)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--replacing_prompt", type=str, default=None)
    parser.add_argument("--pre_train_mlm", default=False, action='store_true')
    parser.add_argument("--task_transfer_projector", default=False, action='store_true')
    parser.add_argument("--model_transfer_projector", default=False, action='store_true')
    parser.add_argument("--activate_neuron", default=True, action='store_true')
    parser.add_argument("--mode", type=str, default="valid")
    parser.add_argument("--projector", type=str, default=None)


    args = parser.parse_args()
    configFilePath = args.config


    config = create_config(configFilePath)



    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    #os.system("clear")
    config.set('distributed', 'local_rank', args.local_rank)
    config.set("distributed", "use", False)
    if config.getboolean("distributed", "use") and len(gpu_list)>1:
        torch.cuda.set_device(gpu_list[args.local_rank])
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
        config.set('distributed', 'gpu_num', len(gpu_list))


    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    set_random_seed(args.seed)


    parameters = init_all(config, gpu_list, args.checkpoint, args.mode, local_rank = args.local_rank, args=args)
    do_test = False

    model = parameters["model"]
    valid_dataset = parameters["valid_dataset"]


    ##########################
    ##########################
    '''准备hook'''
    '''这是提取特征的代码'''
    outputs=[[] for _ in range(12)]
    def save_ppt_outputs1_hook(n):
        def fn(_,__,output):
            outputs[n].append(output.detach().to("cpu"))
            #outputs[n].append(output.detach())
        return fn


    for n in range(12):
        model.encoder.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))


    model.eval()
    valid(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode=args.mode, args=args)


    #################################################
    #################################################
    #################################################

    #merge 17 epoch
    for k in range(12):
        outputs[k] = torch.cat(outputs[k])


    outputs = torch.stack(outputs)


    outputs = outputs[:,:,:1,:] #12 layers, [mask]
    print(outputs.shape)
    # [12, 128, 231, 3072] --> 12, 128(eval batcch size), 231(1 or 100), 3072




