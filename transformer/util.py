import math

import torch

'''
transformer这种基础模型 必须像肌肉记忆一样闪电写出
'''

# self-attention计算函数，这里已经考虑了multi-head的情况
def attention(Q, K, V, mask):
    # [b, 50, 32] -> [b, 4, 50, 8] 在输出这个QKV之前，这样就成功将维度变成了我们当前想要的维度。一个head的QKV对应一个二维的张量
    #【b, 50, 32】的意思是每一句话都有50个token，每一个token对应3维度的向量
    # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
    # Q K V是
    # Q,K,V = [b, 4, 50, 8]
    #如果每一个token是8维度的话，这里假定每一个token的维度为【2,4】
    #因此和当前的Wq相乘，可以得到这个Q的值，Wq的大小是【32,32】，然后经过变化得到 [b, 50, 32] -> [b, 4, 50, 8]

    # [b, 4, 50, 8] * [b, 4, 8, 50] -> [b, 4, 50, 50]
    # Q,K矩阵相乘,求每个词相对其他所有词的注意力，只交换了K矩阵后面的两个轴，其实这里可以使用transpose
    # 每一个头产生了50*50的score，每一个token对应一个分数，那应该是【b，4,50,1】啊，为什么是【b，4,50,50】呢？
    # 因为每一个注意力分数的score，都会得到和自己以及另外的其他全部token之间的注意力分数。有5个token，那么一个token则有5个分数。
    # 这里有50个token，那么每一个token则对应50个score，这样才方便后续计算softmax
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))

    # 除以每个头维数的平方根,做数值缩放
    score /= 8 ** 0.5

    # mask遮盖,mask是true的地方都被替换成-inf,这样在计算softmax的时候,-inf会被压缩到0，exp(x)就是softmax的公式。当x趋近于负无穷时候，exp(X)就约等于0.
    # mask = [b, 1, 50, 50]
    # 需要了解一下mask fill函数的用法，以及mask送入进来的形状：形状应该就是提前计算好的带有true的矩阵
    score = score.masked_fill_(mask, -float('inf'))
    print("当前的score为：",score)
    #用了softmax才会得到真正在softmax之后的分数score，-1表示最后一个维度
    score = torch.softmax(score, dim=-1)

    # 以注意力分数乘以V,得到最终的注意力结果
    # [b, 4, 50, 50] * [b, 4, 50, 8] -> [b, 4, 50, 8]
    score = torch.matmul(score, V)

    # 每个头计算的结果合一,把所有头计算的结果拼接在一起即可
    # [b, 4, 50, 8] -> [b, 50, 32]
    score = score.permute(0, 2, 1, 3).reshape(-1, 50, 32)

    return score


# 多头注意力计算层，多头注意力当中已经包含了对attention函数的使用
class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #这里就是当前我们的Wq/k/v，这个就是用来缩放的W矩阵
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)

        self.out_fc = torch.nn.Linear(32, 32)

        # 规范化之后,均值是0,标准差是1
        # BN是取不同样本做归一化
        # LN是取不同通道做归一化
        # affine=True,elementwise_affine=True,指定规范化后,再计算一个线性映射
        # norm = torch.nn.BatchNorm1d(num_features=4, affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
        """
        [[[-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047],
         [-1.1761, -1.0523, -0.9285, -0.8047]],

        [[ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761],
         [ 0.8047,  0.9285,  1.0523,  1.1761]]]"""

        # norm = torch.nn.LayerNorm(normalized_shape=4, elementwise_affine=True)
        # print(norm(torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)))
        """
        [[[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]],

        [[-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416],
         [-1.3416, -0.4472,  0.4472,  1.3416]]]"""

        self.norm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)

        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask):
        # b句话,每句话50个词,每个词编码成32维向量，同时，我们的QKV的大小也正好是32维。
        # Q,K,V = [b, 50, 32]
        # b，得到我们的batch数
        b = Q.shape[0]

        # 保留下原始的Q,后面要做短接用
        clone_Q = Q.clone()

        # 规范化，规范化之后再做线性运算，这样会让实验的效果更好
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        # 线性运算,维度不变，经过WQ、K、V矩阵
        # [b, 50, 32] -> [b, 50, 32]
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        # 拆分成多个头，拆分的时候可以不用考虑物理意义，可以随意拆分
        # b句话,每句话50个词,每个词编码成32维向量,4个头,每个头分到8维向量
        # [b, 50, 32] -> [b, 4, 50, 8]
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)

        # 拆分了之后就可以计算注意力
        # [b, 4, 50, 8] -> [b, 50, 32]，每一个头计算出来的结果占据8个维度，之类已经把4个头的计算结果纵向拼接在了一起。
        score = attention(Q, K, V, mask)

        # 计算输出,维度不变，得到score之后，我们还通过了看了线性层进行投射，投射完成后再过了一个dropout层。
        #一个token对应这32维，32维分别为z1-z4组合而成。每一个z都是8维
        # 每一个z的维度应该都和QKV的维度相同，每一个token对应32的维度，那么50个token就对应50*32的维度。
        # [b, 50, 32] -> [b, 50, 32]
        score = self.dropout(self.out_fc(score))

        # 短接
        score = clone_Q + score
        return score


# 位置编码层
class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # pos是第几个词,i是第几个维度,d_model是维度总数
        def get_pe(pos, i, d_model):
            fenmu = 1e4 ** (i / d_model)
            pe = pos / fenmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        # 初始化位置编码矩阵，50行32列，每一句话50个词，每一个token对应着32维的向量
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(0)

        # 定义为不更新的常量
        # register butter就是用来定义不进行更新的常亮的 在这里进行申明
        self.register_buffer('pe', pe)

        # 词编码层
        self.embed = torch.nn.Embedding(39, 32)
        # 初始化参数
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # [8, 50] -> [8, 50, 32]
        embed = self.embed(x)

        # 词编码和位置编码相加（词编码矩阵在这里是可以做梯度下降的，而位置编码不需要，我们同时在这里定义了词编码和位置编码）
        # [8, 50, 32] + [1, 50, 32] -> [8, 50, 32] 这里做了一次数据的广播
        embed = embed + self.pe
        return embed


# 全连接输出层,本质上就只做了一个经过全连接的跳级连接，而没有self-attention的跳级连接，感觉不太对
class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)

    def forward(self, x):
        # 保留下原始的x,后面要做短接用
        # 克隆了一份数据，使用clone函数
        # 这个层相当于自己通过全连接神经网络然后连接和没有通过全连接神经网络的自己连接
        clone_x = x.clone()

        # 规范化
        x = self.norm(x)

        # 线性全连接运算
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        # 做短接
        out = clone_x + out

        return out


if __name__ == "__main__":
    Q=torch.randn(3, 4, 50, 8)
    K=torch.randn(3, 4, 50, 8)
    V=torch.randn(3, 4, 50, 8)
    mask=torch.randn(3, 1, 50, 50)
    attention = attention(Q,K,V,mask)