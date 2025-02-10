import torch
from torch import nn
from torch.nn import functional as F

# 输入图片的维度为：torch.Size([32, 3, 32, 32])
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2,padding=0),
        )
        #卷积神经网络的部分完成了，现在开始是flatten层，我们可以自定义flatten层，也可以直接使用自带的linear层

        self.fc_unit=nn.Sequential(
            #这里将卷积cnn输出的维度进行汇聚
            nn.Linear(32*5*5, 2400),
            nn.ReLU(),
            nn.Linear(in_features=2400, out_features=1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=10)
        )

        tmp=torch.randn(2,3,32,32)
        out=self.conv_unit(tmp)
        #卷积之后的输出维度为：【b，16,5,5】
        print('conv out',out.shape)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size = x.size(0)
        x=self.conv_unit(x)
        x=x.view(batch_size,-1) #需要把数据的dimension转化为能够让那个fc unit神经网络能够识别的形状
        logits=self.fc_unit(x)

        # logits的维度为【b，10】，因此我们是在index=1的维度上做的softmax，而不是0维度，这个batch的维度上
       # pred=F.softmax(logits,dim=1)

        return logits




net=LeNet5()

