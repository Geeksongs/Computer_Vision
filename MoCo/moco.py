import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.models import resnet50


transform=ToTensor()
dataset=CIFAR10(root='./cifat10',train=True,download=True,transform=transform)
loader=DataLoader(dataset,batch_size=64,shuffle=True)

def get_resnet50(output_dim):
    model=resnet50(pretrained=True)
    print(model) #model最后的输出维度取到了2048
    model.fc=nn.Linear(2048,output_dim)
    return model


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

C=1024
N=loader.batch_size
K=4096



f_q=get_resnet50(C).to(device)
f_k=get_resnet50(C).to(device)#已经知道加载的是哪一个网络了

#f_q.state_dict()是fq当中拿到的网络当中的参数

print("打印出当前 f_q 的参数 shape:")
for name, param in f_q.state_dict().items():
    print(f"{name}: {param.shape}")

'''
这是fq的参数：
fc.weight: torch.Size([1024, 2048])
fc.bias: torch.Size([1024])
'''

print("\n打印出当前 f_k 的参数 shape:")
for name, param in f_k.state_dict().items():
    print(f"{name}: {param.shape}")


'''
这是fk的参数：
fc.weight: torch.Size([4096, 2048])
fc.bias: torch.Size([4096])

'''

f_k.load_state_dict(f_q.state_dict())#加载网络当中的参数

queue=torch.randn(C,K).to(device)

queue_ptr=0
m=0.99
optimizer=torch.optim.Adam(f_k.parameters(),lr=1e-3)

#随便使用一个手段我们进行数据增强
def aug(x):
    return x+0.1*torch.randn_like(x)

#定义一个loss
def info_nce_loss(q,k,queue,temperature=0.07):
    q=nn.functional.normalize(q,dim=1,p=2)
    k=nn.functional.normalize(k,dim=1,p=2)
    queue=nn.functional.normalize(queue,dim=0,p=2)

    positive_similarity=torch.bmm(q.view(N,1,C),k.view(N,C,1))
    negative_similarity=torch.mm(q,queue)

    #这里还把最后一个维度为1的维度给挤压掉了
    logits=torch.cat((positive_similarity.squeeze(-1),negative_similarity),dim=1)

    labels=torch.zeros(N,dtype=torch.long).to(device)

    loss=nn.CrossEntropyLoss()(logits/temperature,labels)

    return loss


for i,(x,_) in enumerate(loader):
    x=x.to(device)
    x_q=aug(x)
    x_k=aug(x)

    #这里经过了resnet50这个网络
    q=f_q(x_q)
    k=f_k(x_k)

    loss=info_nce_loss(q,k,queue)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("第{}次迭代，当前的loss是{}".format(i,loss.item()))



    with torch.no_grad():
        for param_q,param_k in zip(f_q.parameters(),f_k.parameters()):
            param_k.data=param_k.data*m+param_q.data*(1-m)

        #在当前的一轮迭代当中

        batch_size=k.size(0)
        queue[:,queue_ptr:queue_ptr+batch_size]=k.T
        #循环差值，应该是利用了队列的性质，之前的数据结构里学到过
        queue_ptr = (queue_ptr+batch_size)%K










