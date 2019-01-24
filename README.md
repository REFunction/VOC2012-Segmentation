<p align="center">
<strong>English</strong> | <a href="https://github.com/REFunction/VOC2012-Segmentation/blob/master/README_zh.md"><strong>中文</strong></a>
</p>

# VOC2012-Segmentation
This code will help you use Pascal VOC 2012 Dataset to do research on Semantic Segmentation.
# Update
I have finished a slim version of VOC2012.py, **VOC2012_slim.py**. You don't need to preprocess any thing. And you can learn to use it very easily. It seems no problems but I think it's safe to keep these two versions for a period of time.

Usage:Put VOC2012_slim.py into your project and run it, set the path as yours.
## Prerequisition
- python 3
- cv2
- numpy
- PIL
- h5py
## I want to train my network with VOC2012 dataset
This code will help you finish your semantic image segmentation task. Follow these steps.
1. Download Pascal VOC 2012 dataset
https://pan.baidu.com/s/1L_H66mV1cnhOCm4ACypHZg
Download and extract. It contains train、validation、augmentation dataset without other content.
Download h5 file of augmentation dataset if your computer doesn't have so much memory.
2. Create VOC2012 object
``` python
voc2012 = VOC2012('./VOC2012/')
```
Only to assign the root path.

3. Read all data and save as .h5 form
h5py is an excellent package which can save and load image data very quickly. You don't need to read raw images every time before training.
``` python
voc2012.read_all_data_and_save()
```
This line helps read training and validation datasets and save into './voc2012_train.h5' and './voc2012_val.h5'. You can alse set the path anywhere you want.
At least 16 GB memory will be needed if you want to save augmentation dataset as .h5 form.
``` python
voc2012.read_aug_images_labels_and_save()
```
Or you can download voc2012_aug.h5 directly.

4. Load h5 file
You can load into memory fast next time after saving h5 files.
``` python
voc2012.load_all_data()
```
5. Use data
You can easily access training images、training labels、validation images、validation labels like this.
``` python
voc2012.train_images
voc2012.train_labels
voc2012.val_images
voc2012.val_labels
```
All image variables have the shape [None, self.image_size, self.image_size, 3]
All label variables have the shape [None, self.iamge_size, self.image_size]

6. Get batch during training
``` python
batch_train_images, batch_train_labels = voc2012.get_batch_train(batch_size=8)
batch_val_images, batch_val_labels = voc2012.get_batch_val(batch_size=8)
```
Much faster if you are using augmentation dataset like this.
``` python
batch_images, batch_labels = voc2012.get_batch_aug_fast(batch_size=8)
```
## I want to pre-train with COCO dataset as the same 20 classes with VOC
Ok, follow the below.
1. Download COCO2014 semantic segmentation dataset here https://pan.baidu.com/s/1jrAwWYI-IW35_L4b3nr5dA
  You can also combine training and validation dataset into one.
2. Put COCO2014.py into your project and use like this
``` python
coco2014 = COCO2014('./COCO/val2014/images/', './COCO/val2014/annotations/')
while training:
  image_batch, label_batch = coco2014.get_batch_fast(batch_size=8)
```
Note: I recommand you to use get_batch_fast() while get_batch() can be also used. The former use another threading and a queue to read data while you call sees.run(). Very efficient.
## But I cannot use and understand this code
Don't worry. You can solve your problems by
1. Raise an issue
2. Send me an email function@bupt.edu.cn or 806675223@qq.com
3. QQ:806675223
