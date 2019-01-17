<p align="center">
<strong>English</strong> | <a href="https://github.com/REFunction/VOC2012-Segmentation/master/README_zh.md"><strong>中文</strong></a>
</p>
# VOC2012-Segmentation
This code will help you use Pascal VOC 2012 Dataset to do research on Semantic Segmentation.
## Prerequisition
- python 3
- cv2
- numpy
- PIL
- h5py
## How to Start
To use Pascal VOC 2012 Dataset, you must have one.

http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

Download and extract it.

### Now create a VOC object like this.

``` python
voc2012 = VOC2012('./VOC2012/')
```
Only to assign the root path of your dataset.Note that the path parameter refers to the layer with name 'VOC2012'.

### Read all data and save in the form of h5
h5py is an excellent packge which can save and load images very fast.You don't need to read raw images every time before you train your model.
``` python
voc2012.read_all_data_and_save()
```
Then it will read both train and validation data and save into './voc2012_train.h5' and './voc2012_val.h5'.You can change the locations.
### Load h5
After saving .h5 files, you can load them next time at a very fast speed like this.
``` python
voc2012.load_all_data()
```
## How to use
You can get train images, train labels, validation images, validation labels simply by this.
``` python
voc2012.train_images
voc2012.train_labels
voc2012.val_images
voc2012.val_labels
```
All images variables' shapes are like [None, self.image_size, self.image_size, 3]

All labels variables' shapes are like [None, self.iamge_size, self.image_size]

## How to use during training a network
You can get a batch just like this.
``` python
batch_train_images, batch_train_labels = voc2012.get_batch_train(batch_size=8)
batch_val_images, batch_val_labels = voc2012.get_batch_val(batch_size=8)
```
## About Size
There are 2 methods you can choose to set the uniform image size.
They are 'resize' and 'pad'.
'resize' means uses cv2.resize() to set the image size you want, while 'pad' means add zeros at the bottom and right if the original image is smaller than 500x500.
So when you set 'pad' method, all images and labels will be padding into 500x500.
