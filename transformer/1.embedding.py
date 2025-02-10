import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import copy
import matplotlib.pyplot as plt

print("finish!")

#1.normal embedding
# 这种embedding是自带的，第一个参数是embedding的number，第二个参数是每一个embedding的长度
embedding=nn.Embedding(10,3)
input1=torch.LongTensor([[1,2,9],[1,5,3]])
print(embedding(input1))


#2.padding 0 embedding，只要是idx当中的下标，都输出为0。比如idx=2，那么idx=2的为0
embedding=nn.Embedding(10,3,padding_idx=2)
input1=torch.LongTensor([[0,2,9],[1,0,2]])
print(embedding(input1))


# 3.我们自定义一个embedding
class Embedding(nn.Module):
    def __init__(self,d_model,vocab):
        super(Embedding, self).__init__()
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model

    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)


d_model=512

vocab=1000 #数字超过1000就会报错
x=Variable(torch.LongTensor([[1,2,9,242],[1,5,3,999]]))

emb=Embedding(d_model,vocab)
embr=emb(x)
print()
print(embr)
#最终是一个每一个数字对应这512个维度的向量
print("the shape is ,",embr.shape)


