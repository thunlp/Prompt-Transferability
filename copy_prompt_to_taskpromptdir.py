import os
#import shutil
import shutil
import torch

all_model_prompt = os.listdir("model")

all_model_prompt = [dir for dir in all_model_prompt if ".py" not in dir]


for dataset_file in all_model_prompt:

    #if dataset_file != "tweetevalsentimentPromptRoberta":
    #    continue

    #print(file)

    original_dir = "model/"+str(dataset_file)
    if os.path.isdir(original_dir):
        pass
    else:
        continue

    check_list = [file for file in os.listdir(original_dir) if "_task_prompt" in file]
    if len(check_list) == 0:
        continue

    ##:mean do not use

    ##Choose epoch
    max_epoch = 0


    if dataset_file == "IMDBPromptRoberta":
        max_epoch = 40
    elif dataset_file == "SST2PromptRoberta":
        max_epoch = 18
    elif dataset_file  == "laptopPromptRoberta":
        max_epoch = 32
    elif dataset_file == "restaurantPromptRoberta":
        max_epoch = 32
    elif dataset_file == "movierationalesPromptRoberta":
        max_epoch = 48
    elif dataset_file == "tweetevalsentimentPromptRoberta":
        #max_epoch = 38
        max_epoch = 23

    elif dataset_file == "MNLIPromptRoberta":
        max_epoch = 30 #hv
    elif dataset_file == "QNLIPromptRoberta":
        max_epoch = 67 #hv
    elif dataset_file == "WNLIPromptRoberta":
        max_epoch = 755
    ##elif dataset_file == "anliPromptRoberta":
    ##    max_epoch = 2
    elif dataset_file == "snliPromptRoberta":
        max_epoch = 17 #hv
    elif dataset_file =="RTEPromptRoberta":
        max_epoch = 250


    elif dataset_file == "QQPPromptRoberta":
         max_epoch = 26 #training
    elif dataset_file == "MRPCPromptRoberta":
        max_epoch = 30

    elif dataset_file == "recastfactualityPromptRoberta":
        max_epoch = 21
    elif dataset_file == "recastpunsPromptRoberta":
        max_epoch = 36
    ##elif dataset_file == "recastverbnetPromptRoberta":
    ##    max_epoch =
    elif dataset_file == "recastverbcornerPromptRoberta":
        max_epoch = 35
    elif dataset_file == "recastnerPromptRoberta":
        max_epoch = 18
    elif dataset_file == "recastsentimentPromptRoberta":
        max_epoch = 58
    elif dataset_file == "recastmegaveridicalityPromptRoberta":
        max_epoch = 32
    elif dataset_file == "ethicscommonsensePromptRoberta":
        max_epoch = 96
    elif dataset_file == "ethicsdeontologyPromptRoberta":
        max_epoch = 77
    elif dataset_file == "ethicsjusticePromptRoberta":
        max_epoch = 63
    ##elif dataset_file == "ethicsvirtuePromptRoberta":
    ##    max_epoch = 21


    else:
        for file in os.listdir(original_dir):
            present_epoch = int(file.strip().split("_")[0])
            if present_epoch > max_epoch:
                max_epoch = present_epoch

    original_dir = original_dir+"/"+str(max_epoch)+"_task_prompt.pkl"
    parameters = torch.load(original_dir, map_location=lambda storage, loc: storage)
    prompt_emb = parameters["model"]


    target_dir = "task_prompt_emb"+"/"+str(dataset_file)
    if os.path.isdir(target_dir):
        pass
    else:
        os.mkdir(target_dir)


    target_dir = target_dir+"/"+"task_prompt"

    torch.save(prompt_emb, target_dir)

    print("Save:", target_dir, " Done")

