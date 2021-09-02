import torch
import os
import collections

files = os.listdir()
files = [file for file in files if ".py" not in file]
#print(files)
#exit()

for file in files:
    print(file)


    try:
        all_parameters = torch.load(file+"/15.pkl", map_location=lambda storage, loc: storage)
        #continue
    except:
        print(file,": No suck 15.pkl")
        #os.remove(file+"/task_prompt_emb")
        continue




    #save_orderdict = collections.OrderedDict()
    for key in all_parameters["model"].keys():
        if "embeddings.prompt_embeddings.weight" in key:
            if "roberta" in key:
                prompt_emb = all_parameters["model"]["encoder.roberta.embeddings.prompt_embeddings.weight"]
                #save_orderdict["encoder.roberta.embeddings.prompt_embeddings.weight"] = prompt_emb
            elif "bert" in key:
                prompt_emb = all_parameters["model"]["encoder.bert.embeddings.prompt_embeddings.weight"]
                #save_orderdict["encoder.bert.embeddings.prompt_embeddings.weight"] = prompt_emb
            elif "roberta-large" in key:
                print("roberta-large")
                print("check")
                exit()

    torch.save(prompt_emb, file+"/task_prompt")

