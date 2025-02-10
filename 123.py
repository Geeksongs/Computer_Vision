import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")


#现在开始正式的部分

from torch.utils.data import DataLoader, Dataset
from PIL import Image

# hymenoptera_data/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg

import os
class MyData(Dataset):

    def __init__(self, root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):

        return len(self.img_path)

root_dir="hymenoptera_data/hymenoptera_data/train"
ants_label_dir="ants"
ants_dataset=MyData(root_dir,ants_label_dir)

bees_label_dir="bees"
bees_dataset=MyData(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset

img,label=train_dataset[123]
img.show()


