<p align="center">
<strong>English</strong> | <a href="https://github.com/REFunction/VOC2012-Segmentation/edit/master/README_zh.md"><strong>中文</strong></a>
</p>

# VOC2012-Segmentation
This code will help you use Pascal VOC 2012 Dataset to do research on Semantic Segmentation.
## Prerequisition
- python 3
- cv2
- numpy
- PIL
- h5py
## I want to train my network with Pascal VOC 2012
It's very easy when using this code to help finish your semantic image segmentation work.Follow the following steps.
1. To use Pascal VOC 2012 Dataset, you must have one.
https://pan.baidu.com/s/1L_H66mV1cnhOCm4ACypHZg
Download the dataset from this url.It contains train/val/augmentation dataset.
Download and extract it.
If you don't have much memory for creating augmentation dataset into .h5 file, you can download .h5 files directly.
2. create a VOC object like this.
``` python
voc2012 = VOC2012('./VOC2012/')
```
Only to assign the root path of your dataset.

3. Read all data and save in the form of h5
h5py is an excellent packge which can save and load images very fast.You don't need to read raw images every time before you train your model.
``` python
voc2012.read_all_data_and_save()
```
Then it will read both train and validation data and save into './voc2012_train.h5' and './voc2012_val.h5'.You can change the locations.
If you want to create augmentation dataset into .h5 file, you must have 16 GB memory at least
``` python
voc2012.read_aug_images_labels_and_save()
```
Or you can download voc2012_aug.h5

4. Load h5
After saving .h5 files, you can load them next time at a very fast speed like this.
``` python
voc2012.load_all_data()
```
5. Get data of numpy form
You can get train images, train labels, validation images, validation labels simply by this.
``` python
voc2012.train_images
voc2012.train_labels
voc2012.val_images
voc2012.val_labels
```
All images variables' shapes are like [None, self.image_size, self.image_size, 3]
All labels variables' shapes are like [None, self.iamge_size, self.image_size]

6. Get batch training a network
You can get a batch just like this.
``` python
batch_train_images, batch_train_labels = voc2012.get_batch_train(batch_size=8)
batch_val_images, batch_val_labels = voc2012.get_batch_val(batch_size=8)
```
And for augmentation dataset, there is a much faster function
``` python
batch_images, batch_labels = voc2012.get_batch_aug_fast(batch_size=8)
```
## I want to pretrain my network with coco in 20 classes
Ok. Follow these steps.
1. Download the COCO2014 semantic segmentation here https://pan.baidu.com/s/1jrAwWYI-IW35_L4b3nr5dA
  Extract and you can combine train and validation folder into one.
2. Get COCO2014.py into your project and use like this.
``` python
coco2014 = COCO2014('./COCO/val2014/images/', './COCO/val2014/annotations/')
while training:
  image_batch, label_batch = coco2014.get_batch_fast(batch_size=8)
```
Note: I recommand get_batch_fast() while get_batch() can be alse used. The fromer is implemented with another theading and maintaining a queue, which is much faster when you call sess.run() during training.
## I don't understand how to use and there are so much puzzling code
Ok. You can solve your problems with
1. Raise your issue
2. Send e-mail to me function@bupt.edu.cn or 806675223@qq.com
3. Contact with me with QQ:806675223
