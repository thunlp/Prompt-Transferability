import torch
import os
import sys
import csv

print('Argument List:', sys.argv[1:])

#nli, emotion, sentence_sim
###########################
filter_tokens = sys.argv[1:]
include_mlm = False
#task_define ={"anli":"nli","emobankarousal":"emotion","emobankdominance":"emotion","IMDB":"emotion","laptop":"emotion","MNLI":"nli","movierationales":"emotion","MRPC":"sentence_sim","persuasivenessrelevance":"diseval","persuasivenessspecificity":"diseval","QNLI":"nli","QQP":"sentence_sim","restaurant":"emotion","RTE":"nli","snli":"nli","squinkyformality":"diseval","squinkyimplicature":"diseval","SST2":"emotion","STSB":"other","tweetevalsentiment":"emotion","WNLI":"nli","recastfactuality":"nli","recastmegaveridicality":"nli","recastner":"nli","recastpuns":"nli","recastsentiment":"nli","recastverbcorner":"nli","recastverbnet":"nli","ethicscommonsense":"accept","ethicsdeontology":"accept","ethicsjustice":"accept","ethicsvirtue":"accept"}

#task_define ={"IMDB":"emotion","laptop":"emotion","MNLI":"nli","movierationales":"emotion","MRPC":"sentence_sim","QNLI":"nli","QQP":"sentence_sim","restaurant":"emotion","snli":"nli","SST2":"emotion","tweetevalsentiment":"emotion","WNLI":"nli","ethicsdeontology":"accept","ethicsjustice":"accept"}

task_define ={"IMDB":"emotion","laptop":"emotion","MNLI":"nli","movierationales":"emotion","MRPC":"sentence_sim","QNLI":"nli","QQP":"sentence_sim","restaurant":"emotion","snli":"nli","SST2":"emotion","tweetevalsentiment":"emotion","WNLI":"nli","ethicsdeontology":"ethics","ethicsjustice":"ethics","recastner":"nli"}
###########################



dir = "../task_prompt_emb"
#dir = "../task_prompt_emb_onlyMLMEpoch15Prompt"
files = os.listdir(dir)

project_files = list()
project_files_bert = list()
project_files_roberta = list()

project_emb = list()
project_emb_bert = list()
project_emb_roberta = list()

for file in files:

    if file.replace("PromptRoberta","") not in task_define:
        continue

    if include_mlm == False:
        if "_mlm" in file:
            continue


    for tokens in filter_tokens:
        if tokens in file:
            print(file)

            if "PromptBert" in file:
                task = file.replace("PromptBert","")
                model = "Bert"
            else:
                task = file.replace("PromptRoberta","")
                model = "Roberta"

            if "Bert" in file:
                project_files_bert.append(file+"\t"+task_define[task]+"\t"+model)
            elif "Roberta" in file:
                project_files_roberta.append(file+"\t"+task_define[task]+"\t"+model)

            project_files.append(file+"\t"+task_define[task]+"\t"+model)

            emb = torch.load(dir+"/"+file+"/"+"task_prompt").to("cpu")
            emb = emb.reshape(int(emb.shape[0])*int(emb.shape[1]))
            emb = "\t".join([str(float(e)) for e in emb])


            if "Bert" in file:
                project_emb_bert.append(emb)

            elif "Roberta" in file:
                project_emb_roberta.append(emb)


            project_emb.append(emb)
            #break


print(len(project_files))
print(len(project_emb))

'''
with open(str(",".join(filter_tokens))+'.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(["task_prompt"+"\t"+"task_label"+"\t"+"model"])
    tsv_output.writerow(project_files)


with open(str(",".join(filter_tokens))+'_bert.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(["task_prompt"+"\t"+"task_label"+"\t"+"model"])
    tsv_output.writerow(project_files_bert)
'''


with open(str(",".join(filter_tokens))+'_roberta.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(["task_prompt"+"\t"+"task_label"+"\t"+"model"])
    tsv_output.writerow(project_files_roberta)







'''
with open(str(",".join(filter_tokens)+"_prompt_emb")+'.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(project_emb)


with open(str(",".join(filter_tokens)+"_prompt_emb")+'_bert.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(project_emb_bert)
'''


with open(str(",".join(filter_tokens)+"_prompt_emb")+'_roberta.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(project_emb_roberta)
