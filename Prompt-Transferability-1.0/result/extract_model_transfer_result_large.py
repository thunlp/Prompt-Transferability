import os
import json

all_files = os.listdir()

for file in all_files:
    if "Large" not in file:
        continue

    try:
        #dir = file+"/"+"result_model_transfer_"+file.replace("Roberta","Bert")+".json"
        dir = file+"/"+"result_model_transfer_"+file.replace("Large","")+".json"
        print(file, json.load(open(dir)))
    except:
        continue

