import logging
import torch
from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from .output_init import init_output_function
from torch import nn
from transformers import AutoTokenizer
import string
from tools.projector import AE_0_layer, AE_1_layer

logger = logging.getLogger(__name__)

def trained_AE(load_task_prompt_dir=None):
    PATH="model/crossPromptRoberta/99_model_corss.pkl"
    load_model = torch.load(PATH).to("cuda")
    load_model.eval()


    ####
    #prompt_emb=torch.rand(100,768).to("cuda")
    #prompt_emb=torch.zeros(100,768).to("cuda")
    #prompt_emb=torch.ones(100,768).to("cuda")
    prompt_emb = torch.load(load_task_prompt_dir).to("cuda")
    ####
    cross_prompt_emb = prompt_emb.reshape(int(prompt_emb.shape[0])*int(prompt_emb.shape[1]))

    cross_prompt_emb = load_model(cross_prompt_emb.to("cuda"))
    prompt_emb = cross_prompt_emb.reshape(int(prompt_emb.shape[0]),int(prompt_emb.shape[1]))
    return prompt_emb


'''
class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.encoder = nn.Linear(
            in_features=kwargs["input_dim"], out_features=int(kwargs["input_dim"]/100)
        )
        self.decoder = nn.Linear(
            in_features=int(kwargs["input_dim"]/100), out_features=kwargs["input_dim"]
        )

        # mean-squared error loss
        self.criterion = nn.CrossEntropyLoss()

    def encoding(self, features):
        return self.encoder(features)
    def decoding(self, features):
        return self.decoder(features)


    def forward(self, features):
        encoded_emb = self.encoding(features)
        encoded_emb = torch.relu(encoded_emb)
        decoded_emb = self.decoding(encoded_emb)
        return decoded_emb
'''





def init_all(config, gpu_list, checkpoint, mode, *args, **params):


    result = {}

    logger.info("Begin to initialize dataset and formatter...")
    if mode == "test":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)
    elif mode=="train" or mode=="valid":
        # init_formatter(config, ["test"], *args, **params)
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



    if checkpoint!=None:

        parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        if hasattr(model, 'module'):
            model.module.load_state_dict(parameters["model"])
        else:
            model.load_state_dict(parameters["model"])

        ########################
        ####Evalid will Open####
        ########################
        if "args" in params and mode != "train":
        #if "args" in params:


            #Roberta or Bert
            name_of_model_prompt = string.capwords(params["args"].model_prompt.strip().split("-")[0])

            present_config = params["args"].config
            #Donnot change prompt
            if name_of_model_prompt in present_config:
                pass
            else:
                #Change prompt
                load_task_prompt_dir = params["args"].checkpoint.strip().split("/")[1]


                if "Roberta" in params["args"].checkpoint:
                    #Replace Roberta with Bert
                    load_task_prompt_dir = load_task_prompt_dir.replace("Roberta",name_of_model_prompt)
                    load_task_prompt_dir = "task_prompt_emb/"+load_task_prompt_dir+"/task_prompt"
                elif "Bert" in params["args"].checkpoint:
                    #Replace Bert with Roberta
                    load_task_prompt_dir = load_task_prompt_dir.replace("Bert",name_of_model_prompt)
                    load_task_prompt_dir = "task_prompt_emb/"+load_task_prompt_dir+"/task_prompt"
                else:
                    print("Error")
                    exit()

                prompt_emb = trained_AE(load_task_prompt_dir=load_task_prompt_dir)
                #prompt_emb = torch.nn.Parameter(torch.load(load_task_prompt_dir)).to("cuda")
                #prompt_emb = torch.nn.Parameter(torch.load(load_task_prompt_dir)).to("cuda")
                #print(prompt_emb)
                #prompt_emb = torch.nn.Parameter(torch.load("/data3/private/suyusheng/prompt/prompt/task_prompt_emb/laptopPromptBert/task_prompt")).to("cuda")
                #print(model.encoder.bert.embeddings.prompt_embeddings.weight)
                if "Roberta" in params["args"].checkpoint:
                    model.encoder.roberta.embeddings.prompt_embeddings.weight.data = prompt_emb
                elif "Bert" in params["args"].checkpoint:
                    model.encoder.bert.embeddings.prompt_embeddings.weight.data = prompt_emb
                else:
                    print("Error")
                    exit()

                #exit()
                #model.encoder.bert.embeddings.prompt_embeddings.weight = prompt_emb
        else:
            pass
        ########################
        ########################
        ########################



        if mode == "train":
            trained_epoch = parameters["trained_epoch"]
            if config.get("train", "optimizer") == parameters["optimizer_name"]:
                optimizer.load_state_dict(parameters["optimizer"])
            else:
                logger.warning("Optimizer changed, do not load parameters of optimizer.")

            if "global_step" in parameters:
                global_step = parameters["global_step"]


    #except Exception as e:
    else:
        print("======")
        print("Have no checkpoint")
        print("======")
        #exit()

        #information = "Cannot load checkpoint file with error %s" % str(e)
        #print(information)
        '''
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
