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
from tools.projector import AE_0_layer, AE_1_layer, AE_auto_layer
from transformers import AutoConfig,AutoModelForMaskedLM,AutoTokenizer

logger = logging.getLogger(__name__)

#def recover_model_transfer_prompt(prompt_emb,load_model,projector):
def recover_model_transfer_prompt(prompt_emb,projector,config):
    ##################
    #######AE trained#
    ##################
    '''
    if projector == None:
        if "Bert" in load_model:
            all_model_dir = os.listdir("model/crossPromptBert")
            path = "model/crossPromptBert/"
        elif "Roberta" in load_model:
            all_model_dir = os.listdir("model/crossPromptRoberta")
            path = "model/crossPromptRoberta/"
            print(all_model_dir)

    else:
        all_model_dir = os.listdir(projector)
        print(all_model_dir)
        path = projector+"/"
    '''

    all_model_dir = os.listdir(projector)
    print(all_model_dir)
    path = projector+"/"


    #######################
    #######################
    max_epoch_model=0
    for model in all_model_dir:
        present_epoch_model = int(model.split("_")[0])
        if present_epoch_model > max_epoch_model:
            max_epoch_model = present_epoch_model
            PATH=path+str(model)
    print("===")
    print("Applied Model:",PATH)
    #model = torch.load(PATH).to("cuda")
    #model = torch.load(PATH)
    #model = AE_1_layer(input_dim=76800,compress_dim=768).to("cuda")
    #model = AE_auto_layer(dim_0=768,dim_1=768,dim3=1024).to("cuda")
    if config.get("model","model_size") == "large":
        model = AE_0_layer(dim_0=768,dim_1=1024).to("cuda")
    elif config.get("model","model_size") == "base":
        model = AE_0_layer(dim_0=768,dim_1=768).to("cuda")
    #model = AE_0_layer(dim_0=768,dim_1=768).to("cuda")
    #model = AE_1_layer(dim_0=768,dim_1=int(768/2),dim_2=1024).to("cuda")
    model.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
    #print(model)
    print("===")
    #exit()
    model.eval()
    #######################
    #######################


    #new
    #load_task_prompt_dir = "task_prompt_emb/"+prompt_dir+"/task_prompt"
    #prompt_emb_ = prompt_emb.reshape(int(prompt_emb.shape[0])*int(prompt_emb.shape[1]))
    #prompt_emb_ = torch.nn.Parameter(prompt_emb_)
    prompt_emb_ = model(prompt_emb.to("cuda"))
    #prompt_emb_ = prompt_emb_.reshape(int(prompt_emb.shape[0]),int(prompt_emb.shape[1])).data

    '''
    #load_task_prompt_dir = "task_prompt_emb/"+prompt_dir+"/task_prompt"
    prompt_emb_ = prompt_emb.reshape(int(prompt_emb.shape[0])*int(prompt_emb.shape[1]))
    prompt_emb_ = torch.nn.Parameter(prompt_emb_)
    prompt_emb_ = model(prompt_emb_.to("cuda"))
    prompt_emb_ = prompt_emb_.reshape(int(prompt_emb.shape[0]),int(prompt_emb.shape[1])).data
    '''

    return prompt_emb_





#def recover_task_transfer_prompt(prompt_emb,load_model,projector):
def recover_task_transfer_prompt(prompt_emb,projector):
    ##################
    #######AE trained#
    ##################
    '''
    if projector == None:
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
    else:
        all_model_dir = os.listdir(projector)
        print(all_model_dir)
        path = projector+"/"
    '''

    all_model_dir = os.listdir(projector)
    print(all_model_dir)
    path = projector+"/"

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
    model = torch.load(PATH, map_location=lambda storage, loc: storage).to("cuda")
    model.eval()

    #load_task_prompt_dir = "task_prompt_emb/"+prompt_dir+"/task_prompt"
    prompt_emb_ = prompt_emb.reshape(int(prompt_emb.shape[0])*int(prompt_emb.shape[1]))
    prompt_emb_ = torch.nn.Parameter(prompt_emb_)
    prompt_emb_ = model(prompt_emb_.to("cuda"))
    prompt_emb_ = prompt_emb_.reshape(int(prompt_emb.shape[0]),int(prompt_emb.shape[1])).data

    return prompt_emb_




def init_all(config, gpu_list, checkpoint, mode, *args, **params):

    result = {}


    logger.info("Begin to initialize dataset and formatter...")
    if mode=="test":
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)
    elif mode=="train" or mode=="valid":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
        '''
        print("===================")
        print(result["train_dataset"])
        print(len(result["train_dataset"]))
        print("----")
        print(result["valid_dataset"])
        print(len(result["valid_dataset"]))
        print("===================")
        exit()
        '''
    else:
        print("Don't need to load data")

    logger.info("Begin to initialize models...")

    print(config.get("model", "model_name"))

    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    #print(params) #{'local_rank': -1, 'prompt_emb_output': True}
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    global_step = 0





    #########
    ##########
    if params["args"].checkpoint != None and mode=="train":

        ###################
        model_type = params["args"].checkpoint.split("Prompt")[-1]
        if model_type == "Bert":
            load_dir = "BertForMaskedLM/PromptBert_init_params/pytorch_model.bin"
            if os.path.exists(load_dir):
                parameters = torch.load(load_dir, map_location=lambda storage, loc: storage)
            else:
                print("Not exist:",load_dir)
                exit()
        elif model_type == "Roberta":
            load_dir = "RobertaForMaskedLM/PromptRoberta_init_params/pytorch_model.bin"
            if os.path.exists(load_dir):
                parameters = torch.load(load_dir, map_location=lambda storage, loc: storage)
            else:
                print("Not exist:",load_dir)
                exit()
        elif model_type == "RobertaLarge":
            load_dir = "RobertaLargeForMaskedLM/PromptRobertaLarge_init_params/pytorch_model.bin"
            if os.path.exists(load_dir):
                parameters = torch.load(load_dir, map_location=lambda storage, loc: storage)
            else:
                print("Not exist:",load_dir)
                exit()




        for key in list(parameters):
            parameters["encoder."+key] = parameters.pop(key)


        #print(type(parameters))
        #print(parameters.state_dict())
        #exit()


        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters)
        else:
            model.load_state_dict(parameters)



        '''
        load_checkpoint = params["args"].checkpoint
        files = os.listdir(load_checkpoint)

        if "task_prompt_emb" in files:
            PATH = load_checkpoint+"/task_prompt_emb"
        else:
            max_epoch = 0
            for file in files:
                present_epoch = int(file.split(".")[0])
                if present_epoch > max_epoch:
                    max_epoch = present_epoch
                    PATH=load_checkpoint+"/"+str(max_epoch)+".pkl"
            prompt_parameters = torch.load(PATH, map_location=lambda storage, loc: storage)
            torch.save(prompt_parameters, load_checkpoint+"/task_prompt_emb")
        '''



        prompt_parameters = torch.load(params["args"].replacing_prompt+"/"+"task_prompt", map_location=lambda storage, loc: storage)

        #model
        #optimizer_name
        #optimizer
        #trained_epoch
        #global_step


        #encoder.roberta.embeddings.prompt_embeddings.weight


        if model_type == "Robert" or model_type == "RobertaLarge":
            model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_parameters
        elif model_type == "Bert" or model_type == "BertLarge":
            model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_parameters
        '''
        elif model_type == "RobertLarge":
            model.encoder.roberta_large.embeddings.prompt_embeddings.weight.data = prompt_parameters
            print("check roberta large")
            exit()
        '''



        if torch.cuda.is_available() and mode=="train":
            model.cuda()
        else:
            pass


        ###################
        '''
        parameters = torch.load(params["args"].checkpoint, map_location=lambda storage, loc: storage)
        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters["model"])
        else:
            model.load_state_dict(parameters["model"])

        if torch.cuda.is_available() and mode=="train":
            model.cuda()
        else:
            pass
        '''
        ###################

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
            prompt_name = params["args"].config.split("/")[1].split(".")[0]
            load_task_prompt_dir = "task_prompt_emb/"+prompt_name+"/task_prompt"
            prompt_emb = torch.load(load_task_prompt_dir, map_location=lambda storage, loc: storage)
            print("=========================")
            print("Using",prompt_name,"prompt emb")
            print("=========================")
<<<<<<< HEAD
            '''
            if "Roberta" in prompt_name or "RobertaLarge" in prompt_name:
=======
            config_name = params["args"].config.split("/")[1].split(".")[0]
            #load_task_prompt_dir = "task_prompt_emb/"+config_name+"/task_prompt"
            #prompt_emb = torch.load(load_task_prompt_dir)
            if "Roberta" in config_name or "RobertaLarge" in config_name:
>>>>>>> origin/main
                prompt_emb = model.encoder.roberta.embeddings.prompt_embeddings.weight.data
            elif "Bert" in config_name or "BertLarge" in config_name:
                prompt_emb = model.encoder.bert.embeddings.prompt_embeddings.weight.data
            else:
                print("Warning: Use original prompt emb")
            '''

        elif params["args"].replacing_prompt == "Random" or params["args"].replacing_prompt == "random":
            print("=========================")
            print("Using random prompt emb")
            print("=========================")
            #prompt_emb = torch.nn.Parameter(torch.rand(100,768)).to("cuda")
            config_name = params["args"].config.split("/")[1].split(".")[0]
            if "Large" in config_name:
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),1024).to("cuda")
            else:
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),768).to("cuda")
        else:
            print("=========================")
            print("Replace", params["args"].checkpoint.split("/")[1], "with", params["args"].replacing_prompt)
            print("=========================")
            #load_task_prompt_dir = "task_prompt_emb/"+params["args"].replacing_prompt+"/task_prompt"
            load_task_prompt_dir = params["args"].replacing_prompt+"/task_prompt"
            prompt_emb = torch.load(load_task_prompt_dir, map_location=lambda storage, loc: storage)
        ###

        ###Using Project or not
        if params["args"].task_transfer_projector:
            #load_model = params["args"].checkpoint.strip().split("/")[1]
            #prompt_emb = recover_task_transfer_prompt(prompt_emb,load_model)
            prompt_emb = recover_task_transfer_prompt(prompt_emb,params["args"].projector)
        elif params["args"].model_transfer_projector:
            #load_model = params["args"].checkpoint.strip().split("/")[1]
            #prompt_emb = recover_model_transfer_prompt(prompt_emb,load_model,params["args"].projector)
            prompt_emb = recover_model_transfer_prompt(prompt_emb,params["args"].projector,config)
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
    else:
        print("Mode: Train")
        pass
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

    ############
    if len(gpu_list) > 0:
        if params['local_rank'] < 0:
            model = model.cuda()
        else:
            ###
            #muti machines
            model = model.to(gpu_list[params['local_rank']])

            #single machine
            #model = model.to(params['local_rank'])
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
    ############



    result["model"] = model
    if mode == "train" or mode == "valid":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    logger.info("Initialize done.")


    return result
