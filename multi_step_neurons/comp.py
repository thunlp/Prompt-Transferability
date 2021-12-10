import torch
import os

root = "t5xxl/final"
#root = "t5xxl/wi0"

dirs = os.listdir(root)

# QQP_
#snli_

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
def CosineSimilarity(task1_emb,task2_emb):
    return cos(task1_emb,task2_emb)
    #print(cos(task1_emb,task2_emb))
    #print("=====")


step=1000
for i in range(1,18):

    qqp = torch.load(root+"/"+"QQP_"+str(i*step)+"/neurons.pt").float()
    qqp = qqp.reshape(24, 10240)
    qqp = qqp[21:24,:]
    qqp = qqp.reshape(int(qqp.shape[0])*int(qqp.shape[1]))

    snli = torch.load(root+"/"+"snli_"+str(i*step)+"/neurons.pt").float()
    snli = snli.reshape(24, 10240)
    snli = snli[21:24,:]
    snli = snli.reshape(int(snli.shape[0])*int(snli.shape[1]))


    #qqp[qqp<0] = 0
    #qqp[qqp>0] = 1

    #snli[snli<0] = 0
    #snli[snli>0] = 1


    #print(len(qqp[qqp!=0]), len(snli[snli!=0]))

    print(i, CosineSimilarity(qqp,snli))
    #print(torch.abs(qqp).sum(), torch.abs(snli).sum())
    print("----")
    #print(i)
    #print(qqp)
    #print(qqp.shape)
    #exit()

#print(dirs)
