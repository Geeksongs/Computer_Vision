from torch import nn
import torch
from img_encoder import ImgEncoder
from text_encoder import TextEncoder

#将img encoder和text encoder整合。但是没有经过线性投射层，在这个代码复现当中。
class CLIP(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.img_enc = ImgEncoder()
        self.text_enc = TextEncoder()

    def forward(self, img_x, text_x):
        img_emb = self.img_enc(img_x)
        text_emb = self.text_enc(text_x)
        return img_emb @ text_emb.T


if __name__ == '__main__':
    clip = CLIP()
    img_x = torch.randn(5, 1, 28, 28)
    text_x = torch.randint(0, 10, (5,))
    logits = clip(img_x, text_x)
    print("输出当前logits的shape：",logits.shape)