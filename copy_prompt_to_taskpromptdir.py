import os
#import shutil
import shutil

all_model_prompt = os.listdir("model")

all_model_prompt = [dir for dir in all_model_prompt if ".py" not in dir]

for file in all_model_prompt:
    #print(file)

    original_dir = "model/"+str(file)
    if os.path.isdir(original_dir):
        pass
    else:
        continue

    target_dir = "task_prompt"+"/"+str(file)
    if os.path.isdir(target_dir):
        pass
    else:
        os.mkdir(target_dir)

    target_dir = target_dir+"/"+"task_prompt"
    original_dir = original_dir+"/"+"task_prompt"

    if os.path.isfile(original_dir):
        pass
    else:
        print(file ,"Have to task_prompt")
        continue

    shutil.copyfile(original_dir, target_dir)
