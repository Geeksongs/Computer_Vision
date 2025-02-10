import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lenet5 import LeNet5



def main():
    batch_size = 32

    cifar_train=datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ]
    ),download=True)

    cifar_test=datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ]),download=True)

    cifar_train=DataLoader(cifar_train,batch_size=batch_size,shuffle=True)
    cifar_test=DataLoader(cifar_test,batch_size=batch_size,shuffle=True)

    #使用iter得到一个迭代器，然后使用next得到一个batch
    x,label=next(iter(cifar_train))

    print('x:',x.shape,"label",label.shape)


    #现在开始训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=LeNet5().to(device)
    criterion=nn.CrossEntropyLoss().to(device) #这个当中已经包含了softmax的操作，因此我们不需要写了
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    print(model)

    for epoch in range(30):
        model.train()
        for batch_idx, (x,label) in enumerate(cifar_train):
            x,label=x.to(device),label.to(device)

            logits=model(x)
            loss=criterion(logits,label)

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, loss.item())

        # test部分
        total_correct=0
        total_num=0
        print("start to test")
        # 这一行代码可以告诉pytorch不需要进行反向传播
        model.eval()
        with torch.no_grad():
            for x,label in cifar_test:
                x,label=x.to(device),label.to(device)

                logits=model(x)

                #在dimension=1地方，找到最大的数值，就是我们pred的结果
                pred=logits.argmax(dim=1)
                total_correct += torch.eq(label,pred).float().sum().item()
                total_num += x.size(0)
            acc=total_correct/total_num
            print("当前的epoch为{}，acc为{}".format(epoch,acc))


if __name__ == '__main__':
    main()