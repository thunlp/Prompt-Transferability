import os
#import shutil
import shutil
import torch

all_model_prompt = os.listdir("model")

all_model_prompt = [dir for dir in all_model_prompt if ".py" not in dir]


for dataset_file in all_model_prompt:
    #print(file)

    original_dir = "model/"+str(dataset_file)
    if os.path.isdir(original_dir):
        pass
    else:
        continue

    check_list = [file for file in os.listdir(original_dir) if "_task_prompt" in file]
    if len(check_list) == 0:
        continue


    ##Choose epoch
    max_epoch = 0


    if "IMDBPromptRoberta":
        max_epoch = 72 #hv
    elif "SST2PromptRoberta":
        max_epoch = 48 #hv
    elif "laptopPromptRoberta":
        max_epoch = 48
    elif "restaurantPromptRoberta":
        max_epoch = 48
    elif "movierationalesPromptRoberta":
        max_epoch = 64
    elif "tweetevalsentimentPromptRoberta":
        max_epoch = 32

    #elif "MNLIPromptRoberta":
    #    max_epoch =
    #elif "QNLIPromptRoberta":
    #    max_epoch =
    elif "WNLIPromptRoberta":
        max_epoch = 512
    #elif "anliPromptRoberta":
    #    max_epoch =
    #elif "snliPromptRoberta":
    #    max_epoch =
    elif "RTEPromptRoberta":
        max_epoch = 1000


    #elif "QQPPromptRoberta":
        max_epoch = 52 #training
    elif "MRPCPromptRoberta":
        max_epoch = 48


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

