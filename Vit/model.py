from torch import nn
import torch


class ViT(nn.Module):
    def __init__(self, emb_size=16):
        super().__init__()
        self.patch_size = 4
        self.patch_count = 28 // self.patch_size  # 7

        #这里使用了cnn的方式，很自然地将一张图片切分成一个一个的patch，因为我们设定了stride正好等于patch size的大小。
        #因此每一个卷积核都代表了一个patch。
        #out_channels=4*4=16 ，一共输出16个channels
        #原本的输入为【batch_size,channel=1,width=28,height=28】
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.patch_size ** 2, kernel_size=self.patch_size, padding=0,
                              stride=self.patch_size)  # 图片转patch

        #这个patch embedding就是把patch进行一个线性的投射，上一层经过cnn后，的大小为 (batch_size,channel=16,width=7,height=7)
        #线性层:输入大小：16，输出大小：16，这里相当于linear层没有改变，只是做了一个线性投射（具体需要看论文）
        self.patch_emb = nn.Linear(in_features=self.patch_size ** 2, out_features=emb_size)  # patch做emb
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))  # 分类头输入：分类头的维度：【1，1,16】，这个也是一个课训练参数
        self.pos_emb = nn.Parameter(
            torch.rand(1, self.patch_count ** 2 + 1, emb_size))  # position位置向量 (1,seq_len,emb_size)
        self.tranformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=2, batch_first=True), num_layers=3)  # transformer编码器
        self.cls_linear = nn.Linear(in_features=emb_size, out_features=10)  # 手写数字10分类

    def forward(self, x):  # (batch_size,channel=1,width=28,height=28)
        x = self.conv(x)  # (batch_size,channel=16,width=7,height=7)

        x = x.view(x.size(0), x.size(1), self.patch_count** 2)  # (batch_size,channel=16,seq_len=49)

        #我现在突然有点理解这个轴交换了，轴交换交换的是有意义的数据，即使交换后该轴的意义也不会改变，但是使用view会改变当前轴的意义。该表示batch还是表示batch

        x = x.permute(0, 2, 1)  # (batch_size,seq_len=49,channel=16)

        x = self.patch_emb(x)  # (batch_size,seq_len=49,emb_size)
        # cls_token 的大小为：：【1，1,16】--》【batch_size,1,16】
        cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))  # (batch_size,1,emb_size)
        x = torch.cat((cls_token, x), dim=1)  # add [cls] token
        x = self.pos_emb + x

        y = self.tranformer_enc(x)  # 不涉及padding，所以不需要mask
        return self.cls_linear(y[:, 0, :])  # 对[CLS] token输出做分类


if __name__ == '__main__':
    vit = ViT()
    x = torch.rand(5, 1, 28, 28)
    y = vit(x)
    print(y.shape)

