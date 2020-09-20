# Semantic segmentation(FCN)

**1. Using FCN Model**

​	In this project, I used Tensorflow2.0 and NumPy to built a Fully Convolutional Neural Network(FCN) to do the task that distinguishing the cats and dog’s bodies from the background. Through the training of only one epoch, the accuracy of Semantic Segmentation is reach to 87%, if you have enough GPU resources, you can try to train this model more times, I believe the accuracy can  reach to  about 95%.

**2. The Date which I use is as followed:**

The folder's name of this dataset is :FCN,which contains two folders.

**Annotation:** The data of this date set

**Images:** The images of this dateset 

![微信截图_20200920143559](https://github.com/Geeksongs/Computer_vision/blob/master/Semantic%20segmentation/FCN/123.png)

***If you like my code , please give me a star***,The final visualization is as follows:

![123](https://github.com/Geeksongs/Computer_vision/blob/master/Semantic%20segmentation/FCN/456.png)



**Chinese Translation:**

1.在Using FCN这个项目当中，我使用了Tensorflow2.0，Numpy库来完成了这个项目，如果你想使用我的baseline直接用到其他的地方进行图像语义分割，只需要进行简单修改，更换数据集和像素级的类别个数即可。

在当前模型当中，我采用了全卷积神经网络(FCN)对猫猫狗狗进行图像分割，如果是猫猫或者狗狗，则将其和背景区分开来，整个项目采用了监督学习，我只训练了一个Epoch，图像像素级别的精确度就已经达到了87%，如果你在我模型的基础上继续进行训练的话，我相信你肯定能够达到95%以上。



2.在上述代码当中，我采用了猫狗数据集，数据集的文件夹名称为FCN,里面又有两个文件夹分别是annatation和images，分别用来储存哪些像素点应该具备的分类，images则是所有的图片。你可以打开看看。如果没有数据集，那么说明我没有上传成功，可以使用邮箱freemusic@foxmail.com和我进行联系。



**如果你喜欢我的代码的话，请给我一个Star吧！**
