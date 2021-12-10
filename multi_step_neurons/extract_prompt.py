import os
#import shutil
import shutil
import torch

#all_model_prompt = os.listdir("model")

backbone = "roberta"

all_model_prompt = ["QQPPromptRoberta", "snliPromptRoberta"]

#all_model_prompt = [dir for dir in all_model_prompt if ".py" not in dir]


for dataset_file in all_model_prompt:
    #if "T5" not in dataset_file or "Small" in dataset_file:
    #    continue

    original_dir = "/data/private/suyusheng/prompt/model/"+str(dataset_file)

    check_list = [file for file in os.listdir(original_dir)]


    target_dir = str(backbone)+"/"+str(dataset_file)
    #print(target_dir)
    #exit()
    if os.path.isdir(target_dir):
        pass
    else:
        os.mkdir(target_dir)


    for pkl in check_list:
        target_dir_ = target_dir+"/"
        id_ = pkl.split("_")[0]


        parameters = torch.load(original_dir+"/"+str(pkl), map_location=lambda storage, loc: storage)
        prompt_emb = parameters["model"]

        target_dir_ += "task_prompt_"+str(id_)
        print(target_dir_)
        torch.save(prompt_emb, target_dir_)

    #print("Save:", target_dir, " Done")
    print("Save:",  " Done")
