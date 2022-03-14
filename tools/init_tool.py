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
from tools.projector import AE_0_layer, AE_1_layer_mutiple_100, AE_1_layer, AE_auto_layer
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

    #######################
    #######################
    if ".pkl" in projector or "Random" in projector or "random" in projector:
        print(projector)
        PATH = projector
    else:
        all_model_dir = os.listdir(projector)
        print(all_model_dir)
        path = projector+"/"

        max_epoch_model=0
        for model in all_model_dir:
            present_epoch_model = int(model.split("_")[0])
            if present_epoch_model > max_epoch_model:
                max_epoch_model = present_epoch_model
                PATH=path+str(model)





    print("===")
    print("Applied Model:",PATH)
    model_parameters = torch.load(projector, map_location=lambda storage, loc:storage)
    model = AE_1_layer_mutiple_100(dim_0=int(model_parameters["encoder.weight"].shape[1]),dim_1=int(model_parameters["encoder.weight"].shape[0]),dim_2=int(model_parameters["decoder.weight"].shape[0])).to("cuda")
    #model = torch.load(PATH).to("cuda")
    #model = torch.load(PATH)
    #model = AE_1_layer(input_dim=76800,compress_dim=768).to("cuda")
    #model = AE_auto_layer(dim_0=768,dim_1=768,dim3=1024).to("cuda")
    '''
    if config.get("model","model_size") == "large" and "100" in projector:
        model = AE_1_layer_mutiple_100(dim_0=76800,dim_1=7680,dim_2=102400).to("cuda")
    elif config.get("model","model_size") == "large" and "100" not in projector:
        #model = AE_1_layer(dim_0=768,dim_1=1024).to("cuda")
        model = AE_1_layer(dim_0=768,dim_1=896,dim_2=1024).to("cuda")
    elif config.get("model","model_size") == "base" and "100" in projector:
        #model = AE_0_layer(dim_0=768,dim_1=768).to("cuda")
        #model = AE_0_layer_76800(dim_0=76800,dim_1=76800).to("cuda")
        model = AE_1_layer_mutiple_100(dim_0=76800,dim_1=7680,dim_2=76800).to("cuda")
    elif config.get("model","model_size") == "base" and "100" not in projector:
        #model = AE_1_layer(dim_0=768,dim_1=768,dim_2=768).to("cuda")
        model = AE_0_layer(dim_0=768,dim_1=768).to("cuda")

    #model = AE_0_layer(dim_0=768,dim_1=768).to("cuda")
    #model = AE_1_layer(dim_0=768,dim_1=int(768/2),dim_2=1024).to("cuda")
    '''
    if projector == "Random" or projector=="random":
        pass
    else:
        #model.load_state_dict(torch.load(PATH, map_location=lambda storage, loc: storage))
        model.load_state_dict(model_parameters)

    print("===")
    #exit()
    model.eval()
    #######################
    #######################


    #new
    ####
    if "100" not in projector:
        prompt_emb_ = model(prompt_emb.to("cuda"))

    elif "100" in projector:
        #print(prompt_emb.shape)
        prompt_emb_ = prompt_emb.reshape(1,int(prompt_emb.shape[0])*int(prompt_emb.shape[1]))
        #print(prompt_emb_.shape)
        prompt_emb_ = model(prompt_emb_.to("cuda"))
        #print(prompt_emb_.shape)
        dim_out = int(int(model.decoder.weight.shape[0])/int(prompt_emb.shape[0]))
        prompt_emb_ = prompt_emb_.reshape(int(prompt_emb.shape[0]),dim_out)
        #print(prompt_emb_.shape)
        #exit()
        #prompt_emb_ = prompt_emb_.reshape(int(prompt_emb.shape[0]),int(prompt_emb.shape[1]))
    else:
        print("Wrong: tool/init_tool.py Line:102")
        print(projector)
        exit()


    ####


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
    else:
        print("Don't need to load data")


    logger.info("Begin to initialize models...")

    #if config.get("model", "model_name") == "roberta-small":
    #    model = get_model("RobertaSmallForMaskedLM/"+config.get("model", "model_name"))(config, gpu_list, *args, **params).cuda()
    #    print("====")
    #    print(model)
    #    exit()
    #else:
    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)


    #print("====")
    #print(model)
    #exit()
    #print("====")
    #print(config.get("model", "model_name"))
    #print("====")
    #print(config.get("model", "model_size"))
    #print("====")
    #exit()

    #model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)


    #print(params) #{'local_rank': -1, 'prompt_emb_output': True}
    optimizer = init_optimizer(model, config, *args, **params)
    trained_epoch = 0
    global_step = 0

    #######################
    ###Cross model training
    if os.path.isdir("model/"+config.get("model", "model_name")) and "cross" in config.get("model", "model_name"):
        all_checkpoints = os.listdir("model/"+config.get("model", "model_name"))
        max_checkpoint_epoch = 0
        for checkpoint_epoch in all_checkpoints:
            if int(checkpoint_epoch.split("_")[0]) > max_checkpoint_epoch:
                max_checkpoint_epoch = int(checkpoint_epoch.split("_")[0])
        trained_epoch = max_checkpoint_epoch
    else:
        pass
    #######################

    #print("===============")
    #model_type = params["args"].checkpoint#.split("Prompt")[-1].replace(".config","").replace("_label","")
    #print(model_type)
    #exit()

    ##########
    ##########
    if params["args"].checkpoint != None and mode=="train":
        if os.path.isdir(params["args"].checkpoint) and len(os.listdir(params["args"].checkpoint))!=0:

            ###################
            model_type = params["args"].checkpoint.split("Prompt")[-1].replace(".config","").replace("_label","")

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
            elif model_type == "RobertaSmall":
                load_dir = "RobertaSmallForMaskedLM/PromptRoberta_init_params/pytorch_model.bin"
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
            elif model_type == "T5":
                load_dir = "T5ForMaskedLM/PromptT5_init_params/pytorch_model.bin"
                if os.path.exists(load_dir):
                    parameters = torch.load(load_dir, map_location=lambda storage, loc: storage)
                else:
                    print("Not exist:",load_dir)
                    exit()
            elif model_type == "T5Small":
                load_dir = "T5SmallForMaskedLM/PromptT5Small_init_params/pytorch_model.bin"
                if os.path.exists(load_dir):
                    parameters = torch.load(load_dir, map_location=lambda storage, loc: storage)
                else:
                    print("Not exist:",load_dir)
                    exit()
            elif model_type == "T5Large":
                load_dir = "T5LargeForMaskedLM/PromptT5_init_params/pytorch_model.bin"
                if os.path.exists(load_dir):
                    parameters = torch.load(load_dir, map_location=lambda storage, loc: storage)
                else:
                    print("Not exist:",load_dir)
                    exit()




            for key in list(parameters):
                parameters["encoder."+key] = parameters.pop(key)



            if hasattr(model, 'module'):
                model.module.load_state_dict(parameters)
            else:
                model.load_state_dict(parameters)



            load_checkpoint = params["args"].checkpoint
            files = os.listdir(load_checkpoint)

            #if "task_prompt_emb" in files:
            #    PATH = load_checkpoint+"/task_prompt"
            #else:
            max_epoch = 0
            for file in files:
                present_epoch = int(file.split("_")[0])
                if present_epoch >= max_epoch:
                    max_epoch = present_epoch
                    PATH=load_checkpoint+"/"+str(max_epoch)+"_task_prompt.pkl"
            prompt_parameters = torch.load(PATH, map_location=lambda storage, loc: storage)
            #torch.save(prompt_parameters, load_checkpoint+"/task_prompt_emb")
            #encoder.roberta.embeddings.prompt_embeddings.weight


            if model_type == "Roberta" or model_type == "RobertaLarge":
                model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_parameters["model"]
            elif model_type == "Bert" or model_type == "BertLarge":
                model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_parameters["model"]
            elif model_type == "T5" or model_type == "T5Large" or model_type=="T5Small":
                #model.encoder.t5.embeddings.prompt_embeddings.weight.data = prompt_parameters["model"]
                model.encoder.prompt_embeddings.weight.data = prompt_parameters["model"]
                model.encoder.encoder.prompt_tokens.weight.data = prompt_parameters["model"]
                model.encoder.decoder.prompt_tokens.weight.data = prompt_parameters["model"]
            else:
                print(model_type)
                print("No matching checkpoint load")
                print("init.tool.py Line:273")
                exit()



            if torch.cuda.is_available() and mode=="train":
                model.cuda()
            else:
                pass

            if config.get("train", "optimizer") == prompt_parameters["optimizer_name"]:
                optimizer.load_state_dict(prompt_parameters["optimizer"])
            trained_epoch = prompt_parameters["trained_epoch"]
            global_step = prompt_parameters["global_step"]
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
            '''
            if "Roberta" in prompt_name or "RobertaLarge" in prompt_name:
            config_name = params["args"].config.split("/")[1].split(".")[0]
            #load_task_prompt_dir = "task_prompt_emb/"+config_name+"/task_prompt"
            #prompt_emb = torch.load(load_task_prompt_dir)
            if "Roberta" in config_name or "RobertaLarge" in config_name:
                prompt_emb = model.encoder.roberta.embeddings.prompt_embeddings.weight.data
            elif "Bert" in config_name or "BertLarge" in config_name:
                prompt_emb = model.encoder.bert.embeddings.prompt_embeddings.weight.data
            else:
                print("Warning: Use original prompt emb")
            '''

        elif "Random" in params["args"].replacing_prompt or "random" in params["args"].replacing_prompt and params["args"].replacing_prompt!="randomPromptRobertaLarge":
            print("=========================")
            print("Using random prompt emb")
            print("=========================")
            #prompt_emb = torch.nn.Parameter(torch.rand(100,768)).to("cuda")
            config_name = params["args"].config.split("/")[1].split(".")[0]
            if "Large" in config_name or "large" in config_name:
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),1024).to("cuda")
            if "Small" in config_name:
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),512).to("cuda")
            else:
                prompt_emb = torch.rand(config.getint("prompt","prompt_num"),768).to("cuda")
        else:
            print("=========================")
            print("Replace", params["args"].config.split("/")[1].split(".")[0], "with", params["args"].replacing_prompt)
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
            if "Roberta" in params["args"].config:
                model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
            elif "Bert" in params["args"].config:
                model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
            elif "T5" in params["args"].config:
                #model.encoder.t5.embeddings.prompt_embeddings.weight.data = prompt_emb
                model.encoder.prompt_embeddings.weight.data = prompt_emb
                model.encoder.encoder.prompt_tokens.weight.data = prompt_emb
                model.encoder.decoder.prompt_tokens.weight.data = prompt_emb
            else:
                print("Wrong!!!")
                exit()
        else:
            print("=========================")
            print("Using original prompt emb")
            print("=========================")
            pass

    ########################Can be deleted
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
        elif "T5" in save_name:
            #prompt_emb = model.encoder.t5.embeddings.prompt_embeddings.weight.data
            prompt_emb = model.encoder.prompt_embeddings.weight.data
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


    '''
    try:
        if params["args"].checkpoint==None and mode == "train" or mode == "valid":
            trained_epoch = parameters["trained_epoch"]
            if config.get("train", "optimizer") == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logger.warning("Optimizer changed, do not load parameters of optimizer.")

            if "global_step" in parameters:
                global_step = parameters["global_step"]
    except:
        pass
    '''

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
