import torch

from mask import mask_pad, mask_tril
from util import MultiHead, PositionEmbedding, FullyConnectedOutput


# 编码器层
# encoder当中不需要使用到mask，decoder才需要使用到
# 这里编写的是encoder的上半部分，因此自然跳级连接只跳了一个mlp层
class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()

    def forward(self, x, mask):
        # 输入输出的维度都保持不变
        # 计算自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        # 送入的qkv都是同一个x？
        score = self.mh(x, x, x, mask)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        # self-attention所计算的score输入全连接层，进行跳级连接和layer normalization
        # 这个跳级连接不是这样做的吧，这里应该写错了把。没有写错，只是对encoder下层的另一个跳级连接并没有写而已
        out = self.fc(score)

        return out

# 先写每一层的encoderlayer，再把所有的encoder layer连接起来
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = EncoderLayer()
        self.layer_2 = EncoderLayer()
        self.layer_3 = EncoderLayer()

    def forward(self, x, mask):
        x = self.layer_1(x, mask)
        x = self.layer_2(x, mask)
        x = self.layer_3(x, mask)
        return x


# 解码器层
class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mh1 = MultiHead()
        self.mh2 = MultiHead()

        self.fc = FullyConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        # 先计算y的自注意力,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        # 前面三个参数代表的是QKV
        y = self.mh1(y, y, y, mask_tril_y)

        # 结合x和y的注意力计算,维度不变
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]

        #Decoder这里使用了cross attention，拿到的是encoder当中的输出
        y = self.mh2(y, x, x, mask_pad_x)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 32]
        y = self.fc(y)

        return y


#这是完整的decoder编写
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = DecoderLayer()
        self.layer_2 = DecoderLayer()
        self.layer_3 = DecoderLayer()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer_1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer_3(x, y, mask_pad_x, mask_tril_y)
        return y


# 主模型
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 39)

    def forward(self, x, y):
        #主要搞清楚这个x和y是什么。我们训练的时候拿到的入参变量
        #x就是模型的输入，y就是模型的输出
        # [b, 1, 50, 50]
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)

        # 编码,添加位置信息
        # x = [b, 50] -> [b, 50, 32]
        # y = [b, 50] -> [b, 50, 32]
        x, y = self.embed_x(x), self.embed_y(y)
        #这里塞入的参数x y 还没变成qkv，qkv是在encoder和decoder内部把x和y变成qkv的

        # 编码层计算
        # [b, 50, 32] -> [b, 50, 32]
        x = self.encoder(x, mask_pad_x) #一步就得到了encoder的结果

        # 解码层计算，这里就用到了cross attention，将encoder计算的结果x当做了q和k，然后将预测值y当做了v
        # 这就是cross attention在pytorch当中的使用
        # [b, 50, 32],[b, 50, 32] -> [b, 50, 32]
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)

        # 全连接输出,维度不变
        # [b, 50, 32] -> [b, 50, 39]
        y = self.fc_out(y)

        return y