import torch
b=torch.rand(32)

b=b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)

print(b.squeeze(3).shape)

#2.接下来测试repeat和expand
print(b.expand(4,32,4,8).shape)

print(b.repeat(1,32,4,2).shape)


#3.接下来开始测试t转至矩阵
a=torch.randn(4,32)
print("that is a's shape")
print(a.shape)
print(a.t().shape)

#4.接下来开始对transpose的理解
x=torch.arange(16).reshape(2,2,4)
print()
print(x)
print(x[1,0,2])

print()
x=x.transpose(0,1)
print(x)
print(x[0,1,2])

#[2,1,3]=[1,2,3]

#5.拆分实验
b=torch.randn(32,8)
print(b)
a=torch.randn(32,8)

c=torch.stack([a,b],dim=0)
c=c.repeat(2,1,1)

print(c.shape)

# 现在开始按长度进行拆分
d,e=c.split([2,2],dim=0)
print(d.shape)
print(e.shape)

d,e,f,g=c.chunk(4,dim=0)
print(d.shape)
print(e.shape)


#6.矩阵相乘 乘法实验

