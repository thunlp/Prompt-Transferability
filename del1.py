import torch
a=torch.tensor(
              [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, 10, 1]
              ])
print(a)
print(a.shape)
print(torch.argmax(a,dim=0))
print(torch.argmax(a,dim=1))
#b=torch.argmax(a,dim=1)
#print(a.index_select(1, b))
#print(torch.argmax(a,dim=2))
