import torch
from torch.nn import functional as F
from torch import nn


#像resnet这种基础模块还是的会的，最基本最基本的，复现起来问题不大。

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()

        # 这个resblock可以保证当前的feature map的大小不变，我们只进行channel的变化
        self.conv1=nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(out_channel)

        self.extra=nn.Sequential()
        if out_channel!=in_channel:
            self.extra=nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channel),
                                     )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #element-wise add : [b,ch_in,h,w] with [b,ch_out,h,w]

        #这里就实现了跳级连接，让送入的x与经过卷积的x相加。如果送入的x channel大小和out的不一样，则改变为out的大小。
        #这里没有对送入的x定具体数值的原因是，因为我们有很多resblock，然后我们需要不同的channel in的维度最终都进行shortcut的操作
        # 不然直接使用数字进行变幻就行了
        out=self.extra(x)+out
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        #经过这个卷积层 256大小的图的大小最后还是256
        self.conv1=nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)


        self.bn1=nn.BatchNorm2d(64)

        self.block1=ResBlock(32, 64)
        self.block2=ResBlock(64, 128)
        self.block3=ResBlock(128, 256)
        self.block4=ResBlock(256, 512)
        #【b，c，h，w】==【b，1024,256,256】==》不可能变成【b，512】！
        # 图片大小如果是32，那么就是【b，1024,32,32】
        #block 4接上512？的in feature，不太合理啊
        self.outlayer=nn.Linear(512*32*32, 10)

    def forward(self, x):
        #先经过了一个卷积层
        x=F.relu(self.conv1(x))

        # [b,64,h,w] --> [b,1024,h,w]
        # channel 从64到1024
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=x.view(x.size(0), -1) #第一个维度取出batch size，第二个维度为任意变化
        x = self.outlayer(x)

        return x



#对ResNet的维度进行测试：
model=ResNet18()
tmp=torch.randn(2,3,32,32)
out=model(tmp)
print("resnet:",out.shape)


