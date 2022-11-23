import torch
import os

dir = os.listdir()
for l in dir:
    if ".py" in l:
        continue
    #a = torch.load(l+"/task_prompt").type(torch.float32)
    a = torch.load(l+"/latest.pt").type(torch.float32)
    print(a)
    if int(a.shape[0]) == 100 and int(a.shape[1]) == 4096:
        print(l+"/task_prompt", "Done")
        continue
    else:
        a = a.reshape(int(a.shape[0])*int(a.shape[1]),int(a.shape[2]))
    torch.save(a,l+"/task_prompt")
    print(l+"/task_prompt", "Done")

