import torch
from torch import nn
import math

from torch.autograd import Variable
from torch.nn.functional import dropout
from torch.onnx.symbolic_opset11 import embedding_renorm

embr=1
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000,dropout=0.1):
        super(PositionalEncoding, self).__init__()

        #实例化dropout层
        self.dropout = nn.Dropout(p=dropout)


        #初始化一个位置编码矩阵
        pe= torch.zeros(max_len, d_model)

        #初始化一个绝对位置矩阵
        position=torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        #定义一个变化矩阵div——term，跳跃式的变化
        div_term=torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)))

        #将前面定义的变化矩阵进行奇数偶数分别赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #将二维张量扩充为三维张量
        pe=pe.unsqueeze(0)


        #将位置编码矩阵注册成模型的buffer，这个buffer不是模型中的参数，不跟随优化器同步

        self.register_buffer('pe', pe)
    def forward(self, x):
        #x：代表文本序列的词嵌入表示
        # pe的编码太长了，将第二个维度，也就是max-length对应的维度缩小成x的句子长度
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

d_model=512
dropout=0.1
max_len=60

x=embr
pe=PositionalEncoding(d_model,max_len,dropout)
pe_result=pe(x)
print(pe_result)
print(pe_result.shape)