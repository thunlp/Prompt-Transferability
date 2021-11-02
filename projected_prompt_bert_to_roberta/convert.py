import torch
import os

files = os.listdir()

for file in files:
    #if "IMDB" not in file:
    #    continue
    if ".py" not in file and "task_prompt_emb" not in file:
        print(file)
        a = torch.load(file, map_location=lambda storage, loc: storage)["model"]
        file = file.replace(".pkl","")
        try:
            os.mkdir("task_prompt_emb/"+file+"PromptRoberta")
        except:
            pass
        torch.save(a,"task_prompt_emb/"+file+"PromptRoberta"+"/"+"task_prompt")
        #print("task_prompt_emb/"+file+"/"+"task_prompt")
    else:
        pass
        #print("11111")
exit()


#print(a["model"])
#print(a["model"].shape)
