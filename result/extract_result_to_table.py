import os
import json



#dirs = os.listdir(root_dir)

order_list = ["IMDBPromptRoberta", "SST2PromptRoberta", "laptopPromptRoberta", "restaurantPromptRoberta", "movierationalesPromptRoberta", "tweetevalsentimentPromptRoberta", "MNLIPromptRoberta", "QNLIPromptRoberta", "snliPromptRoberta", "recastnerPromptRoberta", "ethicsdeontologyPromptRoberta","ethicsjusticePromptRoberta","QQPPromptRoberta", "MRPCPromptRoberta", "random"]

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
    p_word = p.replace("PromptRoberta","")
    if len(p_word) > 5:
        p_word = p_word[:5]
    print(p_word, end="\t")
print()




for dataset in order_list:
    #result_dir = root_dir+dataset+"/"

    print_word = dataset.replace("PromptRoberta","")
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

