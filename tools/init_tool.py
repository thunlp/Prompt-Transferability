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

'''
class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        #self.encoder = nn.Linear(in_features=kwargs["input_dim"],out_features=kwargs["compress_dim"])
        self.encoder = nn.Linear(in_features=768*100,out_features=3)
        #self.decoder = nn.Linear(in_features=kwargs["compress_dim"],out_features=kwargs["input_dim"])
        self.decoder = nn.Linear(in_features=3,out_features=768*100)

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)

    def forward(self, features):
        #activation = torch.relu(activation)
        #return reconstructed
        encoded_emb = self.encoding(features)
        decoded_emb = self.decoding(encoded_emb)
        return decoded_emb
'''


def recover_transfer_prompt(prompt_dir):

    ##################
    #######AE trained#
    ##################
    all_model_dir = os.listdir("model/projectPromptRoberta")
    print(all_model_dir)

    max_epoch_model=0
    for model in all_model_dir:
        present_epoch_model = int(model.split("_")[0])
        if present_epoch_model > max_epoch_model:
            max_epoch_model = present_epoch_model
            PATH="model/projectPromptRoberta/"+str(model)
    print("Applied Model:",PATH)
    ###
    #PATH="model/projectPromptRoberta/99_model_AE.pkl"
    ###
    model = torch.load(PATH).to("cuda")
    model.eval()

    load_task_prompt_dir = "task_prompt_emb/"+prompt_dir+"/task_prompt"
    input = torch.nn.Parameter(torch.load(load_task_prompt_dir))
    prompt_emb = input.reshape(int(input.shape[0])*int(input.shape[1]))
    #print(input.shape)
    prompt_emb = model(prompt_emb.to("cuda"))
    #print(recovered_prompt_emb.shape)
    prompt_emb = prompt_emb.reshape(int(input.shape[0]),int(input.shape[1])).data

    return prompt_emb




def init_all(config, gpu_list, checkpoint, mode, *args, **params):
    result = {}

    logger.info("Begin to initialize dataset and formatter...")
    if mode == "train":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
    else:
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)

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
    ####Evalid will Open####
    ########################

    if params["args"].model_transfer:
        #Roberta or Bert
        name_of_model_prompt = string.capwords(params["args"].model_prompt.strip().split("-")[0])
        present_config = params["args"].config
        #Donnot change prompt
        if name_of_model_prompt in present_config:
            pass
        else:
            #Change prompt
            load_task_prompt_dir = params["args"].checkpoint.strip().split("/")[1]

            if "Random" == name_of_model_prompt:
                print("===============")
                print("Random prompt_emb")
                print("===============")
                prompt_emb = torch.nn.Parameter(torch.rand(100,768)).to("cuda")
            elif "Roberta" in params["args"].checkpoint:
                print("===============")
                print("Bert prompt_emb")
                print("===============")
                #Replace Roberta with Bert
                load_task_prompt_dir = load_task_prompt_dir.replace("Roberta",name_of_model_prompt)
                load_task_prompt_dir = "task_prompt_emb/"+load_task_prompt_dir+"/task_prompt"
                prompt_emb = torch.nn.Parameter(torch.load(load_task_prompt_dir)).to("cuda")
            elif "Bert" in params["args"].checkpoint:
                print("===============")
                print("Roberta prompt_emb")
                print("===============")
                #Replace Bert with Roberta
                load_task_prompt_dir = load_task_prompt_dir.replace("Bert",name_of_model_prompt)
                load_task_prompt_dir = "task_prompt_emb/"+load_task_prompt_dir+"/task_prompt"
                prompt_emb = torch.nn.Parameter(torch.load(load_task_prompt_dir)).to("cuda")
            else:
                print("===============")
                print("Error:")
                print("You will use the original prompt from the given model")
                print("===============")
                exit()


            if "Roberta" in params["args"].checkpoint:
                model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
            elif "Bert" in params["args"].checkpoint:
                model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
            else:
                print("Wrong!!!")
                exit()


    elif params["args"].task_transfer:
        load_task_prompt_dir = params["args"].checkpoint.strip().split("/")[1]
        input_emb = recover_transfer_prompt(load_task_prompt_dir)
        prompt_emb = torch.nn.Parameter(input_emb).to("cuda")


        if "Roberta" in params["args"].checkpoint:
            model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
        elif "Bert" in params["args"].checkpoint:
            model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
        else:
            print("Wrong!!!")
            exit()

    else:
        print("Donnot need to change prompt emb")
        pass
    ########################
    ########################
    ########################


    try:
        if mode == "train":
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
    if mode == "train":
        result["optimizer"] = optimizer
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    logger.info("Initialize done.")


    return result
