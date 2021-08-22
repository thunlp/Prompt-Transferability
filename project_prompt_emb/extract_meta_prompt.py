import torch
import os
import sys
import csv

print('Argument List:', sys.argv[1:])


filter_tokens = sys.argv[1:]

dir = "../task_prompt_emb"
files = os.listdir(dir)

project_files = list()
project_emb = list()
for file in files:
    for tokens in filter_tokens:
        if tokens in file:
            project_files.append(file)
            emb = torch.load(dir+"/"+file+"/"+"task_prompt").to("cpu")
            emb = emb.reshape(int(emb.shape[0])*int(emb.shape[1]))
            emb = "\t".join([str(float(e)) for e in emb])
            project_emb.append(emb)
            break

print(project_files)


with open(str(",".join(filter_tokens))+'.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(project_files)


with open(str(",".join(filter_tokens)+"_prompt_emb")+'.tsv', 'w', newline='\n') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\n')
    tsv_output.writerow(project_emb)



