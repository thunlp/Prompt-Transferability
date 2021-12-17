import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import shutil
from timeit import default_timer as timer
import random
import numpy as np
from tools.projector import AE_1_layer_mutiple_100, AE_1_layer_mutiple_100_paper, AE_1_layer_mutiple_100
import torch.nn as nn

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


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def EuclideanDistances(task1_emb,task2_emb):
    return float((1/(1+torch.norm(task1_emb-task2_emb, p='fro'))))


def EuclideanDistances_per_token(task1_emb,task2_emb):
    task1_emb = task1_emb.reshape(100,int(task1_emb.shape[-1]/100))
    task2_emb = task2_emb.reshape(100,int(task2_emb.shape[-1]/100))
    sum_euc = 0
    for idx1, v1 in enumerate(task1_emb):
        #print(idx1)
        #print(v1)
        #print(v1.shape)
        #exit()
        for idx2, v2 in enumerate(task2_emb):
            #euc = torch.norm(v1-v2, p='fro')
            euc = torch.norm(v1-v2, p=2)
            #print(euc)
            sum_euc += euc
    return float((1/float(1+(float(sum_euc/100)/100))))
    #return torch.norm(task1_emb-task2_emb, p='fro')


cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    #return cos(task1_emb,task2_emb).sum()
    return float(cos(task1_emb,task2_emb))

def CosineSimilarity_per_token(task1_emb,task2_emb):
    #print(int(task1_emb.shape[-1]/100))
    task1_emb = task1_emb.reshape(100,int(task1_emb.shape[-1]/100))
    task2_emb = task2_emb.reshape(100,int(task2_emb.shape[-1]/100))
    sum_c = 0
    #return cos(task1_emb,task2_emb).sum()
    for idx1, v1 in enumerate(task1_emb):
        for idx2, v2 in enumerate(task2_emb):
            c = cos(v1,v2)
            sum_c += c
    return float(float(sum_c/100)/100)


def ActivatedNeurons(activated_1, activated_2, layer, backbone_model=None):

    activated_1 = activated_1.float()
    activated_2 = activated_2.float()

    if "T5XXL" in backbone_model:
        if layer == 24:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1])

            activated_2 = activated_2[int(layer):int(layer)+3,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1])

    else:
        if layer ==12:
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])
        else:
            activated_1 = activated_1[int(layer):int(layer)+3,:,:,:]
            activated_1 = activated_1.reshape(activated_1.shape[0]*activated_1.shape[1]*activated_1.shape[2]*activated_1.shape[3])

            activated_2 = activated_2[int(layer):int(layer)+3,:,:,:]
            activated_2 = activated_2.reshape(activated_2.shape[0]*activated_2.shape[1]*activated_2.shape[2]*activated_2.shape[3])


    activated_1[activated_1>0] = float(1)
    activated_1[activated_1<0] = float(0)

    activated_2[activated_2>0] = float(1)
    activated_2[activated_2<0] = float(0)

    sim = cos(activated_1, activated_2)

    return float(sim)





logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)



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

    ###########################
    ###########################
    projector = args.projector
    dataset_org = args.replacing_prompt
    dataset = args.replacing_prompt
    #print(projector)
    #print(args.replacing_prompt)
    #print(config.get("args","replacing_prompt"))
    #config.set("distributed", "use", False)
    #exit()
    to = "T5_to_Roberta"
    model_parameters = torch.load(projector, map_location=lambda storage, loc:storage)

    model_AE = AE_1_layer_mutiple_100(dim_0=int(model_parameters["encoder.weight"].shape[1]),dim_1=int(model_parameters["encoder.weight"].shape[0]),dim_2=int(model_parameters["decoder.weight"].shape[0])).to("cpu")
    model_AE.load_state_dict(model_parameters)


    #prompt = torch.load("task_prompt_emb/"+str(dataset)+"Prompt"+str(base_model)+"/task_prompt", map_location=lambda storage, loc:storage)
    prompt = torch.load(dataset+"/task_prompt", map_location=lambda storage, loc:storage)

    prompt = prompt.reshape(1,int(model_parameters["encoder.weight"].shape[1]))
    p_prompt = torch.Tensor(model_AE(prompt))
    p_prompt = p_prompt.reshape(100, int(p_prompt.shape[-1]/100))

    task_prompt_emb_dir, dataset = dataset.split("/")
    dataset, m = dataset.split("Prompt")

    proj_prompt_dir = "task_prompt_emb/"+str(dataset)+"Prompt"+"_cross_model_"+str(to)
    try:
        os.mkdir(proj_prompt_dir)
    except:
        pass
    torch.save(p_prompt , proj_prompt_dir+"/task_prompt")

    args.replacing_prompt = proj_prompt_dir
    #config.set("replacing_prompt", save_dir)
    #exit()
    ###########################
    ###########################


    parameters = init_all(config, gpu_list, args.checkpoint, args.mode, local_rank = args.local_rank, args=args)
    do_test = False

    model = parameters["model"]
    valid_dataset = parameters["valid_dataset"]




    outputs=[[] for _ in range(12)]
    def save_ppt_outputs1_hook(n):
        def fn(_,__,output):
            outputs[n].append(output.detach().to("cpu"))
            #outputs[n].append(output.detach())
        return fn


    for n in range(12):
        model.encoder.roberta.encoder.layer[n].intermediate.register_forward_hook(save_ppt_outputs1_hook(n))


    '''将数据通过模型'''
    '''hook会自动将中间层的激活储存在outputs中'''
    model.eval()
    valid(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode=args.mode, args=args)




    #merge 17 epoch
    for k in range(12):
        outputs[k] = torch.cat(outputs[k])

    outputs = torch.stack(outputs)

    outputs = outputs[:,:,:1,:] #12 layers, [mask]

    #print(outputs.shape)



    save_name = args.replacing_prompt.strip().split("/")[-1].split(".")[0]+"_"+str(to)
    dir = "task_activated_neuron"
    if os.path.isdir(dir):
        save_dir = dir+"/"+save_name
        if os.path.isdir(save_dir):
            torch.save(outputs,save_dir+"/task_activated_neuron")
        else:
            os.mkdir(save_dir)
            torch.save(outputs,save_dir+"/task_activated_neuron")
    else:
        os.mkdir(dir)
        save_dir = dir+"/"+save_name
        os.mkdir(save_dir)
        torch.save(outputs,save_dir+"/task_activated_neuron")


    #print("==Prompt emb==")
    #print(outputs.shape)
    #print("Save Done")
    #print("==============")


    #exit()
    #################################################
    #################################################
    #################################################

    print("==============")
    print("Caculate sim")
    print("==============")

    ###############
    ###############

    to_model = to.split("_")[-1]

    t_dir = str(dataset)+"Prompt"+to_model

    prompt_org_target = torch.load("task_prompt_emb/"+t_dir+"/task_prompt", map_location=lambda storage, loc:storage)
    prompt_target = torch.load(proj_prompt_dir+"/task_prompt", map_location=lambda storage, loc:storage)
    prompt_org_target = prompt_org_target.reshape(76800)
    prompt_target = prompt_target.reshape(76800)


    prompt_org_target_act = torch.load("task_activated_neuron/"+t_dir+"/task_activated_neuron", map_location=lambda storage, loc:storage)
    prompt_target_act = torch.load(save_dir+"/task_activated_neuron", map_location=lambda storage, loc:storage)
    prompt_org_target_act = prompt_org_target_act[:,0:1,:,:]
    prompt_target_act = prompt_target_act[:,0:1,:,:]

    backbone_model="Roberta"

    print(dataset)
    print("------")
    print("same_task_list_cos:",CosineSimilarity(prompt_org_target,prompt_target))
    print("same_task_list_cos_per_token:",CosineSimilarity_per_token(prompt_org_target,prompt_target))
    print("same_task_list_euc:",EuclideanDistances(prompt_org_target,prompt_target))
    print("same_task_list_euc_per_token:",EuclideanDistances_per_token(prompt_org_target,prompt_target))
    print("same_task_list_neurons_0:",ActivatedNeurons(prompt_org_target_act, prompt_target_act, 0, backbone_model))
    print("same_task_list_neurons_3:",ActivatedNeurons(prompt_org_target_act, prompt_target_act, 3, backbone_model))
    print("same_task_list_neurons_6:",ActivatedNeurons(prompt_org_target_act, prompt_target_act, 6, backbone_model))
    print("same_task_list_neurons_9:",ActivatedNeurons(prompt_org_target_act, prompt_target_act, 9, backbone_model))
    print("same_task_list_neurons_12:",ActivatedNeurons(prompt_org_target_act, prompt_target_act, 12, backbone_model))
    print("======================")





#######





#EuclideanDistances(prompt, p_prompt)
#EuclideanDistances_per_token(prompt, p_prompt)
#CosineSimilarity(prompt, p_prompt)
#CosineSimilarity_per_token(prompt, p_prompt)
#ActivatedNeurons(prompt_neuron, p_prompt_neuron)

