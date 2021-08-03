import logging
import torch
from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from .output_init import init_output_function
from torch import nn
from transformers import AutoTokenizer
import string
import os

logger = logging.getLogger(__name__)

def recover_model_transfer_prompt(prompt_emb,load_model):
    ##################
    #######AE trained#
    ##################
    if "Bert" in load_model:
        all_model_dir = os.listdir("model/crossPromptRoberta")
        path = "model/crossPromptRoberta/"
        print(all_model_dir)
    elif "Roberta" in load_model:
        all_model_dir = os.listdir("model/crossPromptBert")
        path = "model/crosstPromptBert/"
        print(all_model_dir)
    else:
        print("Error in init_tool.py/recover_model_transfer_prompt")


    max_epoch_model=0
    for model in all_model_dir:
        present_epoch_model = int(model.split("_")[0])
        if present_epoch_model > max_epoch_model:
            max_epoch_model = present_epoch_model
            PATH=path+str(model)
    print("Applied Model:",PATH)
    ###
    #PATH="model/projectPromptRoberta/99_model_AE.pkl"
    ###
    model = torch.load(PATH).to("cuda")
    model.eval()

    #load_task_prompt_dir = "task_prompt_emb/"+prompt_dir+"/task_prompt"
    input = torch.nn.Parameter(prompt_emb)
    prompt_emb = input.reshape(int(input.shape[0])*int(input.shape[1]))
    #print(input.shape)
    prompt_emb = model(prompt_emb.to("cuda"))
    #print(recovered_prompt_emb.shape)
    prompt_emb = prompt_emb.reshape(int(input.shape[0]),int(input.shape[1])).data

    return prompt_emb




def recover_task_transfer_prompt(prompt_emb,load_model):
    ##################
    #######AE trained#
    ##################
    if "Bert" in load_model:
        all_model_dir = os.listdir("model/projectPromptBert")
        path = "model/projectPromptBert/"
        print(all_model_dir)
    elif "Roberta" in load_model:
        all_model_dir = os.listdir("model/projectPromptRoberta")
        path = "model/projectPromptRoberta/"
        print(all_model_dir)
    else:
        print("Error in init_tool.py/recover_task_transfer_prompt")

    #all_model_dir = os.listdir("model/projectPromptRoberta")
    #print(all_model_dir)

    max_epoch_model=0
    for model in all_model_dir:
        present_epoch_model = int(model.split("_")[0])
        if present_epoch_model > max_epoch_model:
            max_epoch_model = present_epoch_model
            PATH=path+str(model)
    print("Applied Model:",PATH)
    ###
    #PATH="model/projectPromptRoberta/99_model_AE.pkl"
    ###
    model = torch.load(PATH).to("cuda")
    model.eval()

    #load_task_prompt_dir = "task_prompt_emb/"+prompt_dir+"/task_prompt"
    input = torch.nn.Parameter(prompt_emb)
    prompt_emb = input.reshape(int(input.shape[0])*int(input.shape[1]))
    #print(input.shape)
    prompt_emb = model(prompt_emb.to("cuda"))
    #print(recovered_prompt_emb.shape)
    prompt_emb = prompt_emb.reshape(int(input.shape[0]),int(input.shape[1])).data

    return prompt_emb




def init_all(config, gpu_list, checkpoint, mode, *args, **params):

    result = {}

    logger.info("Begin to initialize dataset and formatter...")
    if mode=="test":
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)
    else:
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)

    logger.info("Begin to initialize models...")

    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    #print(params) #{'local_rank': -1, 'prompt_emb_output': True}
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


    #########
    #try:
    ##########
    #parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    if params["args"].checkpoint != None:
        parameters = torch.load(params["args"].checkpoint, map_location=lambda storage, loc: storage)
        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters["model"])
        else:
            model.load_state_dict(parameters["model"])
    else:
        pass


    ########################
    ########################
    ########################

    ########################
    ####Evalid will Open####
    ########################
    if mode=="valid" or mode=="Valid" or mode=="test" or mode=="Test":
        print("=========================")
        print(params)
        print("=========================")
        ###Replace or not
        if params["args"].replacing_prompt == None:
            print("=========================")
            print("Using original prompt emb")
            print("=========================")
            prompt_emb = None
            pass
        elif params["args"].replacing_prompt == "Random" or params["args"].replacing_prompt == "random":
            print("=========================")
            print("Using random prompt emb")
            print("=========================")
            #prompt_emb = torch.nn.Parameter(torch.rand(100,768)).to("cuda")
            prompt_emb = torch.rand(100,768).to("cuda")
        else:
            print("=========================")
            print("Replace", params["args"].checkpoint.split("/")[1], "with", params["args"].replacing_prompt)
            print("=========================")
            load_task_prompt_dir = "task_prompt_emb/"+params["args"].replacing_prompt+"/task_prompt"
            prompt_emb = torch.load(load_task_prompt_dir)
        ###

        ###Using Project or not
        if params["args"].task_transfer_projector:
            load_model = params["args"].checkpoint.strip().split("/")[1]
            prompt_emb = recover_task_transfer_prompt(prompt_emb,load_model)
        elif params["args"].model_transfer_projector:
            load_model = params["args"].checkpoint.strip().split("/")[1]
            prompt_emb = recover_model_transfer_prompt(prompt_emb,load_model)
        elif params["args"].model_transfer_projector and params["args"].task_transfer_projector:
            print("init_tool.py: Cannot choose both task_project and model_project")
        else:
            print("No project")
            pass

        ##Put prompt emb back to model
        if prompt_emb != None:
            prompt_emb = torch.nn.Parameter(prompt_emb).to("cuda")

            ##Put prompt emb back to model
            if "Roberta" in params["args"].checkpoint:
                model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
            elif "Bert" in params["args"].checkpoint:
                model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
            else:
                print("Wrong!!!")
                exit()
        else:
            print("=========================")
            print("Using original prompt emb")
            print("=========================")
            pass

    ########################
    #Return and Save prompt#
    ########################
    elif mode=="extract_prompt":
        print("=========================")
        print("Extract prompt emb")
        print("=========================")
        #mlm or not mlm
        save_name = params["args"].checkpoint.split("/")[1]

        if "Roberta" in save_name:
            prompt_emb = model.encoder.roberta.embeddings.prompt_embeddings.weight.data
        elif "Bert" in save_name:
            prompt_emb = model.encoder.bert.embeddings.prompt_embeddings.weight.data
        else:
            print("Wrong!!!")

        fp = str("task_prompt_emb/"+save_name)
        if os.path.exists(fp):
            print("Exist:",fp)
        else:
            os.mkdir(fp)
            print("Create:",fp)


        fp_dir = fp+"/task_prompt"
        print("save to:", fp_dir)
        torch.save(prompt_emb, fp_dir)
        print("!!!!!!!")
        print(prompt_emb.shape)
        print("!!!!!!!")
        print("Save prompt_emb_output")
        exit()


    ########################
    ####Train####
    ########################

    ########################
    ########################
    ########################


    try:
        if mode == "train" or mode == "valid":
            trained_epoch = parameters["trained_epoch"]
            if config.get("train", "optimizer") == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logger.warning("Optimizer changed, do not load parameters of optimizer.")

            if "global_step" in parameters:
                global_step = parameters["global_step"]
    except:
        pass

    ###


    '''
    except Exception as e:

        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            logger.error(information)
            raise e
        else:
            logger.warning(information)
    '''



    result["model"] = model
    if mode == "train" or mode == "valid":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    logger.info("Initialize done.")


    return result
