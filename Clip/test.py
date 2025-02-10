# 这个不是测试集 只是为了用来测试torch的功能,模拟实验交叉熵在clip当中的实现

import numpy as np
import torch
import torch.nn.functional as F


#这个输出的还真是一维向量，和
targets = torch.arange(0, 10) #tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(targets)

#logits的shape为【10,10】
logits=torch.rand(10,10)
print(logits)

loss_i = F.cross_entropy(logits, targets)
loss_t = F.cross_entropy(logits.permute(1, 0), targets)

loss = (loss_i + loss_t) / 2

print(loss)











