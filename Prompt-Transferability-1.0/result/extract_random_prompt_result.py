import os
import json

all_files = os.listdir()

for file in all_files:

    try:
        dir = file+"/"+"result_random.json"
        #print(dir)
        #dir = file+"/"+"result_model_transfer_"+file.replace("Large","")+".json"
        print(file,json.load(open(dir)))
    except:
        continue

