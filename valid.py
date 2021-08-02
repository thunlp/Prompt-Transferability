import argparse
import os
import torch
import logging
import random
import numpy as np

from tools.init_tool import init_all
from config_parser import create_config
from tools.valid_tool import valid

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--checkpoint', help="checkpoint file path", type=str, default=None)
    parser.add_argument('--comment', help="checkpoint file path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pre_train_mlm", type=bool, default=False)
    parser.add_argument("--prompt_emb_output", type=bool, default=False)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--task_transfer", type=str, default=False)
    parser.add_argument("--model_transfer", type=str, default=False)


    args = parser.parse_args()

    ####################
    if args.pre_train_mlm == True:
        formatter = "mlmPrompt"
        config.set("data","train_formatter_type",formatter)
        config.set("data","valid_formatter_type",formatter)
        config.set("data","test_formatter_type",formatter)
        config.set("model","model_name","mlmPrompt")
    elif args.task_transfer == True:
        config.set("model","model_name", "projectPromptRoberta_prompt")
    #elif args.model_transfer == True:
    #    config.set("model","model_name", "projectPromptRoberta_prompt")
    else:
        pass


    ####################


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

    os.system("clear")
    config.set('distributed', 'local_rank', args.local_rank)
    if config.getboolean("distributed", "use"):
        torch.cuda.set_device(gpu_list[args.local_rank])
        torch.distributed.init_process_group(backend=config.get("distributed", "backend"))
        config.set('distributed', 'gpu_num', len(gpu_list))

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError
    set_random_seed(args.seed)

    parameters = init_all(config, gpu_list, args.checkpoint, "train", local_rank = args.local_rank, args=args)
    #parameters = init_all(config, gpu_list, args.checkpoint, "train", local_rank = args.local_rank, prompt_emb_output=True)

    #print(parameters)

    do_test = False
    if args.do_test:
        do_test = True

    model = parameters["model"]

    #valid(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode="valid", prompt_emb_output=False, save_name=args.config)
    valid(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode="valid", args=args)
