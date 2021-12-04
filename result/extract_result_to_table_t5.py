import os
import json



#dirs = os.listdir(root_dir)

#order_list = ["IMDBPromptT5", "SST2PromptT5", "laptopPromptT5", "restaurantPromptT5", "movierationalesPromptT5", "tweetevalsentimentPromptT5", "MNLIPromptT5", "QNLIPromptT5", "snliPromptT5", "recastnerPromptT5", "ethicsdeontologyPromptT5","ethicsjusticePromptT5","QQPPromptT5", "MRPCPromptT5", "random"]
order_list = ["IMDBPromptT5", "SST2PromptT5", "laptopPromptT5", "restaurantPromptT5", "movierationalesPromptT5", "tweetevalsentimentPromptT5", "MNLIPromptT5", "QNLIPromptT5", "snliPromptT5", "ethicsdeontologyPromptT5","ethicsjusticePromptT5","QQPPromptT5", "MRPCPromptT5","squadPromptT5","nq_openPromptT5", "multi_newsPromptT5", "samsumPromptT5", "randomPromptT5"]

#root_dir = "result/"

print("Dataset: File dir")
print("|")
print("|")
print("V")


print()
print()

print("Prompt")
print("- - - >")

print()
print()


print(end="\t")
for p in order_list:
    p_word = p.replace("PromptT5","")
    if len(p_word) > 5:
        p_word = p_word[:5]
    print(p_word, end="\t")
print()




for dataset in order_list:
    if "randomPromptT5" in dataset:
        continue
    #result_dir = root_dir+dataset+"/"

    print_word = dataset.replace("PromptT5","")
    if len(print_word) > 5:
        print_word = print_word[:5]
    print(print_word, end="\t")

    result_dir = dataset+"/"
    for prompt in order_list:

        prompt_dir = result_dir+"result_"+prompt+".json"
        try:
            result = round(json.loads(json.load(open(prompt_dir,"r")))["acc"]*100,1)
            print(result, end="\t")
            #print(type(result))
            #print(result)
            #print(json.load(open(result_dir+prompt,"r"))["acc"])
        except:
            print("----", end="\t")
    print()

