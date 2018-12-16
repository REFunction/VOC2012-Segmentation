import cv2
import numpy as np
from PIL import Image
import h5py
import os

def save_h5(path,images,labels):
    print('saving', path)
    file = h5py.File(name=path,mode='w')
    file['images'] = images
    file['labels'] = labels
def load_h5(path):
	print('loading',path)
	file = h5py.File(name=path,mode='r')
	return file['images'],file['labels']


class VOC2012:
    def __init__(self, root_path='./VOC2012/', aug_path='SegmentationClassAug/', image_size=(224, 224),
                 resize_method='resize', checkpaths=False):
        '''
        Create a VOC2012 object
        This function will set all paths needed, do not set them mannully expect you have
        changed the dictionary structure
        Args:
            root_path:the Pascal VOC 2012 folder path
            aug_path:The augmentation dataset path. If you don't want to use it, ignore
            image_size:resize images and labels into this size
            resize_method:'resize' or 'pad', if pad, images and labels will be paded into 500x500
                        and the parameter image_size will not be used
        '''
        self.root_path = root_path
        self.resize_method = resize_method
        if resize_method != 'resize' and resize_method != 'pad':
            print('Unknown resize method:', resize_method)
            exit()
        if root_path[len(root_path) - 1] != '/' and root_path[len(root_path) - 1] != '\\':
            self.root_path += '/'
        self.train_list_path = self.root_path + 'ImageSets/Segmentation/train.txt'
        self.val_list_path = self.root_path + 'ImageSets/Segmentation/val.txt'
        self.image_path = self.root_path + 'JPEGImages/'
        self.label_path = self.root_path + 'SegmentationClass/'
        self.aug_path = aug_path
        if aug_path[len(aug_path) - 1] != '/' and aug_path[len(aug_path) - 1] != '\\':
            self.aug_path += '/'
        self.image_size = image_size
        if checkpaths:
            self.check_paths()

    def check_paths(self):
        '''
        check all paths and display the status of paths
        '''
        if not (os.path.exists(self.root_path) and os.path.isdir(self.root_path)):
            print('Warning: Dictionary', self.root_path, ' does not exist')
        if not (os.path.exists(self.train_list_path) and os.path.isfile(self.train_list_path)):
            print('Warning: Training list file', self.train_list_path, 'does not exist')
        if not (os.path.exists(self.val_list_path) and os.path.isfile(self.val_list_path)):
            print('Warning: Validation list file', self.val_list_path, 'does not exist')
        if not (os.path.exists(self.image_path) and os.path.isdir(self.image_path)):
            print('Warning: Dictionary', self.image_path, 'does not exist')
        if not (os.path.exists(self.label_path) and os.path.isdir(self.label_path)):
            print('Warning: Dictionary', self.label_path, 'does not exist')
        if not (os.path.exists(self.aug_path) and os.path.isdir(self.aug_path)):
            print('Warning: Dictionary', self.aug_path, 'does not exist')

    def read_train_list(self):
        '''
        Read the filenames of training images and labels into self.train_list
        '''
        self.train_list = []
        f = open(self.train_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.train_list.append(line)
        f.close()
    def read_val_list(self):
        '''
        Read the filenames of validation images and labels into self.val_list
        '''
        self.val_list = []
        f = open(self.val_list_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.val_list.append(line)
        f.close()

    def read_train_images(self):
        '''
        Read training images into self.train_images
        If you haven't called self.read_train_list(), it will call first
        After reading images, it will resize them
        '''
        self.train_images = []
        if hasattr(self, 'train_list') == False:
            self.read_train_list()
        for filename in self.train_list:
            image = cv2.imread(self.image_path + filename + '.jpg')
            if self.resize_method == 'resize':
                image = cv2.resize(image, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv2.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv2.BORDER_CONSTANT, value=0)
            self.train_images.append(image)
            if len(self.train_images) % 100 == 0:
                print('Reading train images', len(self.train_images), '/', len(self.train_list))
    def read_train_labels(self):
        '''
        Read training labels into self.train_labels
        If you haven't called self.read_train_list(), it will call first
        After reading labels, it will resize them

        Note:image[image > 100] = 0 will remove all white borders in original labels
        '''
        self.train_labels = []
        if hasattr(self, 'train_list') == False:
            self.read_train_list()
        for filename in self.train_list:
            image = Image.open(self.label_path + filename + '.png')
            image = np.array(image)
            if self.resize_method == 'resize':
                image = cv2.resize(image, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv2.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv2.BORDER_CONSTANT, value=0)
            image[image > 100] = 0
            self.train_labels.append(image)
            if len(self.train_labels) % 100 == 0:
                print('Reading train labels', len(self.train_labels), '/', len(self.train_list))
    def read_val_images(self):
        '''
           Read validation images into self.val_images
           If you haven't called self.read_val_list(), it will call first
           After reading images, it will resize them
        '''
        self.val_images = []
        if hasattr(self, 'val_list') == False:
            self.read_val_list()
        for filename in self.val_list:
            image = cv2.imread(self.image_path + filename + '.jpg')
            if self.resize_method == 'resize':
                image = cv2.resize(image, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv2.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv2.BORDER_CONSTANT, value=0)
            self.val_images.append(image)
            if len(self.val_images) % 100 == 0:
                print('Reading val images', len(self.val_images), '/', len(self.val_list))
    def read_val_labels(self):
        '''
           Read validation labels into self.val_labels
           If you haven't called self.read_val_list(), it will call first
           After reading labels, it will resize them

           Note:image[image > 100] = 0 will remove all white borders in original labels
        '''
        self.val_labels = []
        if hasattr(self, 'val_list') == False:
            self.read_val_list()
        for filename in self.val_list:
            image = Image.open(self.label_path + filename + '.png')
            image = np.array(image)
            if self.resize_method == 'resize':
                image = cv2.resize(image, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv2.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv2.BORDER_CONSTANT, value=0)
            image[image > 100] = 0
            self.val_labels.append(image)
            if len(self.val_labels) % 100 == 0:
                print('Reading val labels', len(self.val_labels), '/', len(self.val_list))
    def read_aug_images_labels_and_save(self, save_path='./voc2012_aug.h5'):
        '''
        read augmentation images and labels, and save them in the form of '.h5'
        Note:This function will cost a great amount of memory. Leave enough to call it.
        '''
        is_continue = input('Warning:Reading augmentation files may take up a lot of memory, continue?[y/n]')

        if is_continue != 'y' and is_continue != 'Y':
            return

        if hasattr(self, 'aug_images') == False:
            self.aug_images = []
        if hasattr(self, 'aug_labels') == False:
            self.aug_labels = []
        # check
        if self.aug_path is None or os.path.exists(self.aug_path) == False:
            raise Exception('No augmentation dictionary.Set attribute \'aug_path\' first')
        if self.image_path is None or os.path.exists(self.image_path) == False:
            raise Exception('Cannot find VOC2012 images path.')

        aug_labels_filenames = os.listdir(self.aug_path)

        for label_filename in aug_labels_filenames:
            # read label
            label = cv2.imread(self.aug_path + label_filename, cv2.IMREAD_GRAYSCALE)
            if self.resize_method == 'resize':
                label = cv2.resize(label, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(label)[0]
                width = np.shape(label)[1]
                label = cv2.copyMakeBorder(label, 0, 500 - height, 0, 500 - width, cv2.BORDER_CONSTANT, value=0)
            label[label > 20] = 0
            self.aug_labels.append(label)
            # read image
            image_filename = label_filename.replace('.png','.jpg')
            image = cv2.imread(self.image_path + image_filename)
            if self.resize_method == 'resize':
                image = cv2.resize(image, self.image_size)
            elif self.resize_method == 'pad':
                height = np.shape(image)[0]
                width = np.shape(image)[1]
                image = cv2.copyMakeBorder(image, 0, 500 - height, 0, 500 - width, cv2.BORDER_CONSTANT, value=0)
            self.aug_images.append(image)
            if len(self.aug_labels) % 100 == 0:
                print('Reading augmentation image & label pairs', len(self.aug_labels), '/',
                                                                    len(aug_labels_filenames))
        save_h5(save_path, self.aug_images, self.aug_labels)
    def load_aug_data(self, aug_data_path='./voc2012_aug.h5'):
        self.aug_images, self.aug_labels = load_h5(aug_data_path)
    def save_train_data(self, path='./voc2012_train.h5'):
        '''
        save training images and labels into path in the form of .h5
        Args:
            path:The path you want to save train data into.It must be xxx.h5
        '''
        save_h5(path, self.train_images, self.train_labels)
    def save_val_data(self, path='./voc2012_val.h5'):
        '''
        save validation images and labels into path in the form of .h5
        Args:
            path:The path you want to save train data into.It must be xxx.h5
        '''
        save_h5(path, self.val_images, self.val_labels)
    def read_all_data_and_save(self, train_data_save_path='./voc2012_train.h5', val_data_save_path='./voc2012_val.h5'):
        '''
        Read training and validation data and save them into two .h5 files.
        Args:
            train_data_save_path:The path you want to save training data into.
            val_data_save_path:The path you want to save validation data into.
        '''
        self.read_train_images()
        self.read_train_labels()
        self.read_val_images()
        self.read_val_labels()
        self.save_train_data(train_data_save_path)
        self.save_val_data(val_data_save_path)
    def load_all_data(self, train_data_load_path='./voc2012_train.h5', val_data_load_path='./voc2012_val.h5'):
        '''
        Load training and validation data from .h5 files
        Args:
            train_data_load_path:The training data .h5 file path.
            val_data_load_path:The validation data .h5 file path.
        '''
        self.load_train_data(train_data_load_path)
        self.load_val_data(val_data_load_path)
    def load_train_data(self, path='./voc2012_train.h5'):
        '''
        Load training data from .h5 files
        Args:
            train_data_load_path:The training data .h5 file path.
        '''
        self.train_images, self.train_labels = load_h5(path)
    def load_val_data(self, path='./voc2012_val.h5'):
        '''
        Load validation data from .h5 files
        Args:
            val_data_load_path:The validation data .h5 file path.
        '''
        self.val_images, self.val_labels = load_h5(path)

    def get_batch_train(self, batch_size):
        '''
        Get a batch data from training data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
            batch_size:The number of images or labels returns at a time.
        Return:
            batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
            batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'train_location') == False:
            self.train_location = 0
        end = min(self.train_location + batch_size, len(self.train_images))
        start = self.train_location
        batch_images = self.train_images[start:end]
        batch_labels = self.train_labels[start:end]
        self.train_location = (self.train_location + batch_size) % len(self.train_images)
        if end - start != batch_size:
            batch_images = np.concatenate([batch_images, self.train_images[0:self.train_location]], axis=0)
            batch_labels = np.concatenate([batch_labels, self.train_labels[0:self.train_location]], axis=0)

        return batch_images, batch_labels
    def get_batch_val(self, batch_size):
        '''
        Get a batch data from validation data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
            batch_size:The number of images or labels returns at a time.
        Return:
            batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
            batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'val_location') == False:
            self.val_location = 0
        end = min(self.val_location + batch_size, len(self.val_images))
        start = self.val_location
        batch_images = self.val_images[start:end]
        batch_labels = self.val_labels[start:end]
        self.val_location = (self.val_location + batch_size) % len(self.val_images)
        if end - start != batch_size:
            batch_images = np.concatenate([batch_images, self.val_images[0:self.val_location]], axis=0)
            batch_labels = np.concatenate([batch_labels, self.val_labels[0:self.val_location]], axis=0)

        return batch_images, batch_labels
    def get_batch_aug(self, batch_size):
        '''
        Get a batch data from augmentation data.
        It maintains an internal location variable and get from start to end gradually.
        When it comes into the end, it returns to the start.
        Args:
           batch_size:The number of images or labels returns at a time.
        Return:
           batch_images:A batch of images with shape:[batch_size, image_size, image_size, 3]
           batch_labels:A batch of labels with shape:[batch_size, image_size, image_size]
        '''
        if hasattr(self, 'aug_location') == False:
            self.aug_location = 0
        end = min(self.aug_location + batch_size, len(self.aug_images))
        start = self.aug_location
        batch_images = self.aug_images[start:end]
        batch_labels = self.aug_labels[start:end]
        self.aug_location = (self.aug_location + batch_size) % len(self.aug_images)
        if end - start != batch_size:
            batch_images = np.concatenate([batch_images, self.aug_images[0:self.aug_location]], axis=0)
            batch_labels = np.concatenate([batch_labels, self.aug_labels[0:self.aug_location]], axis=0)

        return batch_images, batch_labels