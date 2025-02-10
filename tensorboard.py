from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

#writer.add_scalar()
#tag:表示图像的title
#scalar_value：图像的y轴
#global_step:图像的x轴

for i in range(100):
    writer.add_scalar("scalar", i, i)

writer.close()

