import os
#import shutil
import shutil
import torch

all_model_prompt = os.listdir("model")

all_model_prompt = [dir for dir in all_model_prompt if ".py" not in dir]


for dataset_file in all_model_prompt:
    if "T5" not in dataset_file:
        continue

    #if dataset_file != "QQPPromptRoberta":
    #    continue


    #if dataset_file != "ethicscommonsensePromptRoberta":
    #if dataset_file != "MRPCPromptRoberta":
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
        max_epoch = 23
    elif dataset_file == "IMDBPromptRoberta_label":
        max_epoch = 40
    elif dataset_file == "IMDBPromptBert":
        max_epoch = 21
    elif dataset_file == "IMDBPromptT5":
        max_epoch = 70

    elif dataset_file == "SST2PromptRoberta":
        max_epoch = 25
    elif dataset_file == "SST2PromptRoberta_label":
        max_epoch = 18
    elif dataset_file == "SST2PromptBert":
        max_epoch = 18
    elif dataset_file == "SST2PromptT5":
        max_epoch = 1

    elif dataset_file  == "laptopPromptRoberta":
        max_epoch = 32
    elif dataset_file  == "laptopPromptRoberta_label":
        max_epoch = 32
    elif dataset_file  == "laptopPromptBert":
        max_epoch = 30
    elif dataset_file  == "laptopPromptT5":
        max_epoch = 1

    elif dataset_file == "restaurantPromptRoberta":
        max_epoch = 33
    elif dataset_file == "restaurantPromptRoberta_label":
        max_epoch = 32
    elif dataset_file == "restaurantPromptBert":
        max_epoch = 31
    elif dataset_file == "restaurantPromptT5":
        max_epoch = 276

    elif dataset_file == "movierationalesPromptRoberta":
        max_epoch = 21
    elif dataset_file == "movierationalesPromptRoberta_label":
        max_epoch = 48
    elif dataset_file == "movierationalesPromptBert":
        max_epoch = 24
    elif dataset_file == "movierationalesPromptT5":
        max_epoch = 197

    elif dataset_file == "tweetevalsentimentPromptRoberta":
        #max_epoch = 38
        max_epoch = 28
    elif dataset_file == "tweetevalsentimentPromptRoberta_label":
        max_epoch = 23
    elif dataset_file == "tweetevalsentimentPromptBert":
        #max_epoch = 38
        max_epoch = 21
    elif dataset_file == "tweetevalsentimentPromptT5":
        max_epoch = 43


    elif dataset_file == "MNLIPromptRoberta":
        max_epoch = 44 #24 #best:105
    elif dataset_file == "MNLIPromptRoberta_label":
        max_epoch = 30
    elif dataset_file == "MNLIPromptBert":
        max_epoch = 34 #29
    elif dataset_file == "MNLIPromptT5":
        max_epoch = 1


    elif dataset_file == "QNLIPromptRoberta":
        max_epoch = 51 #67
    elif dataset_file == "QNLIPromptRoberta_label":
        max_epoch = 67
    elif dataset_file == "QNLIPromptBert":
        max_epoch = 41 #66
    elif dataset_file == "QNLIPromptT5":
        max_epoch = 10

    elif dataset_file == "WNLIPromptRoberta":
        max_epoch = 755
    elif dataset_file == "WNLIPromptRoberta_label":
        max_epoch = 755
    elif dataset_file == "WNLIPromptBert":
        max_epoch = 754
    ##elif dataset_file == "anliPromptRoberta":
    ##    max_epoch = 2

    elif dataset_file == "snliPromptRoberta":
        max_epoch = 29 #replace with 32
    elif dataset_file == "snliPromptRoberta_label":
        max_epoch = 17
    elif dataset_file == "snliPromptBert":
        max_epoch = 32 #16
    elif dataset_file == "snliPromptT5":
        max_epoch = 1


    elif dataset_file =="RTEPromptRoberta":
        max_epoch = 250
    elif dataset_file =="RTEPromptRoberta_label":
        max_epoch = 250
    elif dataset_file =="RTEPromptBert":
        max_epoch = 249


    elif dataset_file == "QQPPromptRoberta":
         max_epoch =  22 #26
    elif dataset_file == "QQPPromptRoberta_label":
         max_epoch = 26
    elif dataset_file == "QQPPromptBert":
         max_epoch = 24 #25
    elif dataset_file == "QQPPromptT5":
         max_epoch = 26

    elif dataset_file == "MRPCPromptRoberta":
        max_epoch = 66 #30
    elif dataset_file == "MRPCPromptRoberta_label":
        max_epoch = 30
    elif dataset_file == "MRPCPromptBert":
        max_epoch = 27 #29
    elif dataset_file == "MRPCPromptT5":
        max_epoch = 199


    elif dataset_file == "recastfactualityPromptRoberta":
        max_epoch = 21
    elif dataset_file == "recastfactualityPromptRoberta_label":
        max_epoch = 21
    elif dataset_file == "recastfactualityPromptBert":
        max_epoch = 20

    elif dataset_file == "recastpunsPromptRoberta":
        max_epoch = 36
    elif dataset_file == "recastpunsPromptRoberta_label":
        max_epoch = 36
    elif dataset_file == "recastpunsPromptBert":
        max_epoch = 35

    elif dataset_file == "recastverbcornerPromptRoberta":
        max_epoch = 35
    elif dataset_file == "recastverbcornerPromptRoberta_label":
        max_epoch = 35
    elif dataset_file == "recastverbcornerPromptBert":
        max_epoch = 34

    elif dataset_file == "recastnerPromptRoberta":
        max_epoch = 30 #19
    elif dataset_file == "recastnerPromptRoberta_label":
        max_epoch = 18
    elif dataset_file == "recastnerPromptBert":
        max_epoch = 20 #14

    elif dataset_file == "recastsentimentPromptRoberta":
        max_epoch = 58
    elif dataset_file == "recastsentimentPromptRoberta_label":
        max_epoch = 58
    elif dataset_file == "recastsentimentPromptBert":
        max_epoch = 57

    elif dataset_file == "recastmegaveridicalityPromptRoberta":
        max_epoch = 32
    elif dataset_file == "recastmegaveridicalityPromptRoberta_label":
        max_epoch = 32
    elif dataset_file == "recastmegaveridicalityPromptBert":
        max_epoch = 31

    elif dataset_file == "ethicscommonsensePromptRoberta":
        max_epoch = 96
    elif dataset_file == "ethicscommonsensePromptRoberta_label":
        max_epoch = 96
    elif dataset_file == "ethicscommonsensePromptBert":
        max_epoch = 95

    elif dataset_file == "ethicsdeontologyPromptRoberta":
        max_epoch = 63
    elif dataset_file == "ethicsdeontologyPromptRoberta_label":
        max_epoch = 77
    elif dataset_file == "ethicsdeontologyPromptBert":
        max_epoch = 14
    elif dataset_file == "ethicsdeontologyPromptT5":
        max_epoch = 1

    elif dataset_file == "ethicsjusticePromptRoberta":
        max_epoch = 29
    elif dataset_file == "ethicsjusticePromptRoberta_label":
        max_epoch = 63
    elif dataset_file == "ethicsjusticePromptBert":
        max_epoch = 15
    elif dataset_file == "ethicsjusticePromptT5":
        max_epoch =
    ##elif dataset_file == "ethicsvirtuePromptRoberta":
    ##    max_epoch = 21


    elif dataset_file == "squadPromptT5":
        max_epoch = 1
    elif dataset_file == "nq_openPromptT5":
        max_epoch = 1
    elif dataset_file == "multi_newsPromptT5":
        max_epoch = 1
    elif dataset_file == "samsumPromptT5":
        max_epoch = 1

    else:
        print("--------------------")
        print("Did not need to genertate this promt_emb:", dataset_file)
        print("--------------------")
        continue
        '''
        for file in os.listdir(original_dir):
            present_epoch = int(file.strip().split("_")[0])
            if present_epoch > max_epoch:
                max_epoch = present_epoch
        '''

    original_dir = original_dir+"/"+str(max_epoch)+"_task_prompt.pkl"



    try:
        parameters = torch.load(original_dir, map_location=lambda storage, loc: storage)
        prompt_emb = parameters["model"]
    except:
        print(dataset_file,"has no trained task_prompt.pkl at epoch",max_epoch)
        continue


    target_dir = "task_prompt_emb"+"/"+str(dataset_file)
    if os.path.isdir(target_dir):
        pass
    else:
        os.mkdir(target_dir)


    target_dir = target_dir+"/"+"task_prompt"

    torch.save(prompt_emb, target_dir)

    print("Save:", target_dir, " Done")

