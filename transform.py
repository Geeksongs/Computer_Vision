from PIL import Image
from torchvision import transforms
import torch
flag = torch.cuda.is_available()
if flag:
    print("CUDA可使用")
else:
    print("CUDA不可用")
    




img_path="hymenoptera_data/hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img=Image.open(img_path)

tensor_trans=transforms.ToTensor()
tensor_img=tensor_trans(img)

print(tensor_img)
