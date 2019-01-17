
<p align="center">
<a href="https://github.com/REFunction/VOC2012-Segmentation/master/README.md"><strong>English</strong> </a>| <strong>中文</strong>
</p>

# VOC2012-分割
这份代码将帮助你使用Pascal VOC 2012数据集来做语义分割方面的研究。
## 准备工作（安装下面的Python包）
- python 3
- cv2
- numpy
- PIL
- h5py
## 我想用VOC2012数据集训练我的神经网络
这份代码能帮你非常简单地完成你的语义图像分割任务，按下面步骤操作
1. 下载Pascal VOC 2012数据集
https://pan.baidu.com/s/1L_H66mV1cnhOCm4ACypHZg
下载后解压，其中包含了训练集、验证集、增强集，并且去掉了无关的数据
如果你的电脑没有那么多内存来创建增强集的h5文件，可以直接从里面下载
2. 创建VOC2012对象
``` python
voc2012 = VOC2012('./VOC2012/')
```
像这样只需指定根目录即可

3. 读取所有的数据并且存成h5格式
h5py是一个非常好的包，它能快速保存和加载图像数据。你不必每次训练模型前读取原始图像。
``` python
voc2012.read_all_data_and_save()
```
这句代码可以读取训练和验证集并保存到'./voc2012_train.h5' 和 './voc2012_val.h5'。你可以自己设置想要的路径。
如果你想把增强集保存成h5文件，最少需要16GB的内存
``` python
voc2012.read_aug_images_labels_and_save()
```
或者你可以直接下载voc2012_aug.h5

4. 加载h5文件
保存好h5文件后，你可以下次非常快地加载到内存里
``` python
voc2012.load_all_data()
```
5. 访问数据
你能轻松访问到训练图片、训练标签、验证集图片、验证集标签
``` python
voc2012.train_images
voc2012.train_labels
voc2012.val_images
voc2012.val_labels
```
所有图片变量的格式是 [None, self.image_size, self.image_size, 3]
所有标签变量的格式是 [None, self.iamge_size, self.image_size]

6. 训练时获取batch
可以这样获取batch
``` python
batch_train_images, batch_train_labels = voc2012.get_batch_train(batch_size=8)
batch_val_images, batch_val_labels = voc2012.get_batch_val(batch_size=8)
```
如果你用的是增强集，这样可以更快
``` python
batch_images, batch_labels = voc2012.get_batch_aug_fast(batch_size=8)
```
## 我想预训练COCO数据集，只选择和VOC一样的20类
好，按下面的来做
1. 从这里下载COCO2014语义分割数据 https://pan.baidu.com/s/1jrAwWYI-IW35_L4b3nr5dA
  解压即可，你还可以把训练集和验证集合并
2. 把COCO2014.py放进你的项目文件，然后这样用
``` python
coco2014 = COCO2014('./COCO/val2014/images/', './COCO/val2014/annotations/')
while training:
  image_batch, label_batch = coco2014.get_batch_fast(batch_size=8)
```
注意：推荐你使用get_batch_fast()，虽然get_batch()也能用但是比较慢。前者用了额外的线程和队列机制保证你的sess.run的同时读取数据，非常高效。
## 我还是不会用，代码看不懂
没关系，你可以解决问题通过
1. 发起一个issue
2. 给我发邮件 function@bupt.edu.cn 或者 806675223@qq.com
3. 加我QQ:806675223
