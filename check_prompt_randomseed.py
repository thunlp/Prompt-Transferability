import argparse
import logging
import random
import numpy as np
import os
import json
import math
import numpy
import glob
import argparse
import torch

from tools.init_tool import init_all
from config_parser import create_config
from tools.create_tool import create

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


#cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(task1_emb,task2_emb).sum()


def EuclideanDistances(task1_emb,task2_emb):
    #return torch.norm(task1_emb-task2_emb, p='fro')
    sum_euclidence=0
    for i in range(len(task1_emb)):
        sum_euclidence += torch.norm(task1_emb[i]-task2_emb[i], p='fro')
    return sum_euclidence


def Euclidean(task1_emb, task2_emb):
    return torch.cdist(task1_emb,task2_emb,p=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--comment', help="checkpoint file path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--return_or_save", type=str, default="save")
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


    #pkl_files = os.listdir(args.checkpoint)
    #print(pkl_files)
    #exit()

    prefiex=8
    extra_ten = dict()
    extra_task_map = dict()

    #for pkl_file in pkl_files:
    for id in range(1,9):

        checkpoint = args.checkpoint+"_"+str(id)+"/15.pkl"
        parameters = init_all(config, gpu_list, checkpoint, "train", local_rank = args.local_rank)

        do_test = False
        if args.do_test:
            do_test = True

        model = parameters["model"]

        '''
        if args.return_or_save == "save":
            create(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode="valid", prompt_emb_output=True, save_name=args.config, return_or_save="save")
        else:
            prompt_emb = create(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode="valid", prompt_emb_output=True, save_name=args.config, return_or_save="return")
        '''

        prompt_emb = create(model, parameters["valid_dataset"], 1, None, config, gpu_list, parameters["output_function"], mode="valid", prompt_emb_output=True, save_name=args.config, return_or_save="return")

        #id = int(pkl_file.replace(".pkl",""))
        task_name = checkpoint.strip().split("/")[1]
        extra_ten[prefiex+id] = prompt_emb
        extra_task_map[prefiex+id] = task_name+"_seed"+str(id)






    '''
    prefiex=8
    sst_extra_ten=dict()
    sst_task_map=dict()
    for i in range(1,15):
        path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta_"+str(i)+"/"
        sst2_file = os.listdir(path)[0]
        sst2_ten = torch.load(path+sst2_file)
        #print(sst2_ten.shape)
        sst_extra_ten[prefiex+i] = sst2_ten
        sst_task_map[prefiex+i] = "sst2_"+str(i)
    '''



    #SST2
    sst2_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/SST2PromptRoberta/"
    sst2_file = os.listdir(path)[0]
    sst2_ten=torch.load(path+sst2_file)
#sst2_ten = sst2_ten.reshape(int(sst2_ten.shape[0]*sst2_ten.shape[1]))
#sst2_ten = sst2_ten[0]
    print(sst2_ten.shape)


    #RTE
    rte_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/RTEPromptRoberta/"
    rte_file = os.listdir(path)[0]
    rte_ten=torch.load(path+rte_file)
    #rte_ten = rte_ten.reshape(int(rte_ten.shape[0]*rte_ten.shape[1]))
    print(rte_ten.shape)


    #RE
    re_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/REPrompt/"
    re_file = os.listdir(path)[0]
    re_ten=torch.load(path+re_file)
    #re_ten = re_ten.reshape(int(re_ten.shape[0]*re_ten.shape[1]))
    print(re_ten.shape)


    #MNLI
    MNLI_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MNLIPromptRoberta/"
    MNLI_file = os.listdir(path)[0]
    MNLI_ten=torch.load(path+MNLI_file)
    #MNLI_ten = MNLI_ten.reshape(int(MNLI_ten.shape[0]*MNLI_ten.shape[1]))
    print(MNLI_ten.shape)


    #MRPC
    MRPC_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/MRPCPromptRoberta/"
    MRPC_file = os.listdir(path)[0]
    MRPC_ten=torch.load(path+MRPC_file)
    #MRPC_ten = MRPC_ten.reshape(int(MRPC_ten.shape[0]*MRPC_ten.shape[1]))
    print(MRPC_ten.shape)



    #QNLI
    QNLI_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QNLIPromptRoberta/"
    QNLI_file = os.listdir(path)[0]
    QNLI_ten=torch.load(path+QNLI_file)
    #QNLI_ten = QNLI_ten.reshape(int(QNLI_ten.shape[0]*QNLI_ten.shape[1]))
    print(QNLI_ten.shape)


    #QQP
    QQP_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/QQPPromptRoberta/"
    QQP_file = os.listdir(path)[0]
    QQP_ten=torch.load(path+QQP_file)
    #QQP_ten = QQP_ten.reshape(int(QQP_ten.shape[0]*QQP_ten.shape[1]))
    print(QQP_ten.shape)


    #WNLI
    WNLI_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/WNLIPromptRoberta/"
    WNLI_file = os.listdir(path)[0]
    WNLI_ten=torch.load(path+WNLI_file)
    #WNLI_ten = WNLI_ten.reshape(int(WNLI_ten.shape[0]*WNLI_ten.shape[1]))
    print(WNLI_ten.shape)


    #STSB
    STSB_ten = list()
    path="/data3/private/suyusheng/prompt/prompt/task_prompt_emb/STSBPromptRoberta/"
    STSB_file = os.listdir(path)[0]
    STSB_ten=torch.load(path+STSB_file)
    #STSB_ten = STSB_ten.reshape(int(STSB_ten.shape[0]*STSB_ten.shape[1]))
    print(STSB_ten.shape)

    ###########################
    ###########################
    ###########################


    task_ten= {0:sst2_ten,1:rte_ten,2:re_ten,3:MNLI_ten,4:MRPC_ten,5:QNLI_ten,6:QQP_ten,7:WNLI_ten,8:STSB_ten}
    task_ten.update(extra_ten)

#task_ten={0:sst2_ten,1:rte_ten,2:re_ten,3:MNLI_ten,4:MRPC_ten,5:QNLI_ten,6:QQP_ten,7:WNLI_ten,8:STSB_ten,9:sst2_ten_5,10:sst2_ten_10,11:sst2_ten_11,12:sst2_ten_12,13:sst2_ten_13,14:sst2_ten_14}

    task_map={0:"sst2",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB"}
    task_map.update(extra_task_map)
#task_map={0:"sst2_15",1:"rte",2:"re",3:"MNLI",4:"MRPC",5:"QNLI",6:"QQP",7:"WNLI",8:"STSB",9:"sst2_5",10:"sst2_10",11:"sst2_11",12:"sst2_12",13:"sst2_13",14:"sst2_14"}


    for id_1, task_1 in task_map.items():
        cos_dict=dict()
        euc_dict=dict()
        for id_2, task_2 in task_map.items():
            if id_1 == id_2:
                continue
            else:
                #similiarty:
                #cos:
                cos_dict[task_map[id_2]]=float(CosineSimilarity(task_ten[id_1],task_ten[id_2]))

                #endcli
                euc_dict[task_map[id_2]]=float(EuclideanDistances(task_ten[id_1],task_ten[id_2]))

        #ranking
        print("=======================")
        print("==",task_1,"==")
        print("-------")
        print("CosineSimilarity:")
        print("-------")
        for task_2 in sorted(cos_dict, key=cos_dict.get, reverse=True):
            print(task_2, cos_dict[task_2])

        print("-------")
        print("EuclideanDistances:")
        print("-------")
        for task_2 in sorted(euc_dict, key=euc_dict.get, reverse=False):
            print(task_2, euc_dict[task_2])

        print("=======================")



