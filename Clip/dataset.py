from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor, Compose
import torchvision


# 手写数字
class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.ds = torchvision.datasets.MNIST('./mnist/', train=is_train, download=True)
        self.img_convert = Compose([
            PILToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        #在convert之后除以255
        return self.img_convert(img) / 255.0, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = MNIST()
    img, label = ds[0]
    print(label)
    print("This is the current img； ",img.shape) #确实这个minist数据集的图片只有一个channel，令人难以置信！！
    # matplotlib.pyplot.imshow 方法通常期望输入为形状为 (height, width, channels) 的图像数据,因此需要进行轴交换
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
