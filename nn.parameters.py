import torch
from torch import nn


class mod(torch.nn.Module):
    def __init__(self):
        super(mod,self).__init__()
        self.w1=torch.tensor([1,2],dtype=torch.float32,requires_grad=True)
        a=torch.tensor([3,4],dtype=torch.float32)

        self.w2=nn.Parameter(a)

    def forward(self,x):
        o1=torch.dot(self.w1,x)
        o2=torch.dot(self.w2,x)

        return o1+o2

model=mod()
for p in model.parameters():
    print(p)

#开始第二个model的尝试：
model.state_dict()
for para in model.parameters():
    print(para)

