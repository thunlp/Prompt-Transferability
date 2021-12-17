import os
import json



#dirs = os.listdir(root_dir)

#order_list = ["IMDBPromptT5Small", "SST2PromptT5Small", "laptopPromptT5Small", "restaurantPromptT5Small", "movierationalesPromptT5Small", "tweetevalsentimentPromptT5Small", "MNLIPromptT5Small", "QNLIPromptT5Small", "snliPromptT5Small", "recastnerPromptT5Small", "ethicsdeontologyPromptT5Small","ethicsjusticePromptT5Small","QQPPromptT5Small", "MRPCPromptT5Small", "random"]
order_list = ["IMDBPromptT5Small", "SST2PromptT5Small", "laptopPromptT5Small", "restaurantPromptT5Small", "movierationalesPromptT5Small", "tweetevalsentimentPromptT5Small", "MNLIPromptT5Small", "QNLIPromptT5Small", "snliPromptT5Small", "ethicsdeontologyPromptT5Small","ethicsjusticePromptT5Small","QQPPromptT5Small", "MRPCPromptT5Small","squadPromptT5Small","nq_openPromptT5Small", "multi_newsPromptT5Small", "samsumPromptT5Small", "randomPromptT5Small"]
#order_list = ["IMDBPromptT5Small", "SST2PromptT5Small", "laptopPromptT5Small", "restaurantPromptT5Small", "movierationalesPromptT5Small", "tweetevalsentimentPromptT5Small"]

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
    p_word = p.replace("PromptT5Small","")
    if len(p_word) > 5:
        p_word = p_word[:5]
    print(p_word, end="\t")
print()




for dataset in order_list:
    if "randomPromptT5Small" in dataset:
        continue
    #result_dir = root_dir+dataset+"/"

    print_word = dataset.replace("PromptT5Small","")
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

