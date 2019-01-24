import cv2
import numpy as np
from PIL import Image
import h5py
import os
import random
import threading
import queue
import os


class VOC2012:
    def __init__(self, root_path='./VOC2012/', aug_path='SegmentationClassAug/', image_size=(224, 224),
                 resize_method='resize'):
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
        self.train_names_path = self.root_path + 'ImageSets/Segmentation/train.txt'
        self.val_names_path = self.root_path + 'ImageSets/Segmentation/val.txt'
        self.image_path = self.root_path + 'JPEGImages/'
        self.label_path = self.root_path + 'SegmentationClass/'
        self.aug_path = aug_path
        if aug_path[len(aug_path) - 1] != '/' and aug_path[len(aug_path) - 1] != '\\':
            self.aug_path += '/'
        self.image_size = image_size
        if os.path.isfile(self.train_names_path):
            self.read_train_names()
        if os.path.isfile(self.val_names_path):
            self.read_val_names()
        if os.path.isdir(self.aug_path):
            self.read_aug_names()

    def read_train_names(self):
        '''
        Read the filenames of training images and labels into self.train_list
        '''
        self.train_names = []
        f = open(self.train_names_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.train_names.append(line)
        f.close()
    def read_val_names(self):
        '''
        Read the filenames of validation images and labels into self.val_list
        '''
        self.val_names = []
        f = open(self.val_names_path, 'r')
        line = None
        while 1:
            line = f.readline().replace('\n', '')
            if line is None or len(line) == 0:
                break
            self.val_names.append(line)
        f.close()
    def read_aug_names(self):
        filenames = os.listdir(self.aug_path)
        self.aug_names = []
        for i in range(len(filenames)):
            self.aug_names.append(filenames[i][:-4])


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
        end = min(self.train_location + batch_size, len(self.train_names))
        start = self.train_location
        batch_images_names = self.train_names[start:end]
        batch_labels_names = []
        self.train_location = (self.train_location + batch_size) % len(self.train_names)
        if end - start != batch_size:
            batch_images_names = np.concatenate([batch_images_names, self.train_names[0:self.train_location]], axis=0)
        for i in range(batch_size):
            batch_labels_names.append(batch_images_names[i][:-4] + '.png')
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            image = cv2.imread(self.image_path + batch_images_names[i] + '.jpg')
            if self.image_size:
                image = cv2.resize(image, self.image_size)
            label = np.array(Image.open(self.label_path + batch_images_names[i] + '.png'))
            label[label > 20] = 0
            if self.image_size:
                label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            batch_images.append(image)
            batch_labels.append(label)
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
        end = min(self.val_location + batch_size, len(self.val_names))
        start = self.val_location
        batch_images_names = self.val_names[start:end]
        batch_labels_names = []
        self.val_location = (self.val_location + batch_size) % len(self.val_names)
        if end - start != batch_size:
            batch_images_names = np.concatenate([batch_images_names, self.val_names[0:self.val_location]], axis=0)
        for i in range(batch_size):
            batch_labels_names.append(batch_images_names[i][:-4] + '.png')
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            image = cv2.imread(self.image_path + batch_images_names[i] + '.jpg')
            if self.image_size:
                image = cv2.resize(image, self.image_size)
            label = np.array(Image.open(self.label_path + batch_images_names[i] + '.png'))
            label[label > 20] = 0
            if self.image_size:
                label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            batch_images.append(image)
            batch_labels.append(label)
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
        end = min(self.aug_location + batch_size, len(self.aug_names))
        start = self.aug_location
        batch_images_names = self.aug_names[start:end]
        batch_labels_names = []
        self.aug_location = (self.aug_location + batch_size) % len(self.aug_names)
        if end - start != batch_size:
            batch_images_names = np.concatenate([batch_images_names, self.aug_names[0:self.aug_location]], axis=0)
        for i in range(batch_size):
            batch_labels_names.append(batch_images_names[i][:-4] + '.png')
        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            image = cv2.imread(self.image_path + batch_images_names[i] + '.jpg')
            if self.image_size:
                image = cv2.resize(image, self.image_size)
            label = cv2.imread(self.aug_path + batch_images_names[i] + '.png', cv2.IMREAD_GRAYSCALE)
            label[label > 20] = 0
            if self.image_size:
                label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            batch_images.append(image)
            batch_labels.append(label)
        return batch_images, batch_labels
    def add_batch_aug_queue(self, batch_size, max_queue_size, random_resize=True):
        if hasattr(self, 'aug_queue') == False:
            self.aug_queue = queue.Queue(maxsize=max_queue_size)
        while 1:
            image_batch, label_batch = self.get_batch_aug(batch_size)
            if random_resize:
                image_batch, label_batch = self.random_resize(image_batch, label_batch)
            self.aug_queue.put([image_batch, label_batch])
    def start_batch_aug_queue(self, batch_size, max_queue_size=30):
        if hasattr(self, 'aug_queue') == False:
            queue_thread = threading.Thread(target=self.add_batch_aug_queue, args=(batch_size, max_queue_size))
            queue_thread.start()
    def get_batch_aug_fast(self, batch_size, random_resize=True, max_queue_size=30):
        '''
        A fast function for get augmentation batch.Use another thread to get batch and put into a queue.
        :param batch_size: batch size
        :param max_queue_size: the max capacity of the queue
        :param random_resize:If true, the batch will be resized randomly
        :return: An image batch with shape [batch_size, height, width, 3]
                and a label batch with shape [batch_size, height, width, 1]
        '''
        # create queue thread
        if hasattr(self, 'aug_queue') == False:
            queue_thread = threading.Thread(target=self.add_batch_aug_queue, args=(batch_size, max_queue_size, random_resize))
            queue_thread.start()
        while hasattr(self, 'aug_queue') == False:
            time.sleep(0.1)
        image_batch, label_batch = self.aug_queue.get()
        return image_batch, label_batch

    def random_resize(self, image_batch, label_batch, random_blur=False):
        '''
        resize the batch data randomly
        :param image_batch: shape [batch_size, height, width, 3]
        :param label_batch: shape [batch_size, height, width, 1]
        :param random_blur:If true, blur the image randomly with Gaussian Blur method
        :return:
        '''
        new_image_batch = []
        new_label_batch = []
        batch_shape = np.shape(image_batch)
        a = random.random() / 2 + 0.5 # (0,1) -> (0, 1.5)->(0.5, 2)
        b = random.random() / 2 + 0.5 # (0,1) -> (0, 1.5)->(0.5, 2)
        batch_size = batch_shape[0]
        new_height = int(a * batch_shape[1])
        new_width = int(b * batch_shape[2])
        for i in range(batch_size):
            image = image_batch[i]
            if random_blur:
                radius = int(random.randrange(0, 3))  * 2 + 1
                image = cv2.GaussianBlur(image, (radius, radius), random.randrange(0, 3))
            new_image_batch.append(cv2.resize(image, (new_height, new_width)))
            new_label_batch.append(cv2.resize(label_batch[i], (new_height, new_width), interpolation=cv2.INTER_NEAREST))
        return new_image_batch, new_label_batch

    def index_to_rgb(self, index):
        '''
        Find the rgb color with the class index
        :param index:
        :return: A list like [1, 2, 3]
        '''
        color_dict = {0:[0, 0, 0], 1:[128, 0, 0], 2:[0, 128, 0], 3:[128, 128, 0], 4:[0, 0, 128], 5:[128, 0, 128],
                      6:[0, 128, 128], 7:[128, 128, 128], 8:[64, 0, 0], 9:[192, 0, 0], 10:[64, 128, 0],
                      11:[192, 128, 0], 12:[64, 0, 128], 13:[192, 0, 128], 14:[64, 128, 128], 15:[192, 128, 128],
                      16:[0, 64, 0], 17:[128, 64, 0], 18:[0, 192, 0], 19:[128, 192, 0], 20:[0, 64, 128]}
        return color_dict[index]
    @staticmethod
    def gray_to_rgb(image):
        '''
        Convert the gray image(mask image) to a rgb image
        :param image: gray image, with shape [height, width]
        :return: rgb image, with shape [height, width, 3]
        '''
        height = np.shape(image)[0]
        width = np.shape(image)[1]
        result = np.zeros([height, width, 3], dtype='uint8')
        for h in range(height):
            for w in range(width):
                result[h][w] = self.index_to_rgb(image[h][w])
        return result
    def get_one_class_label(self, label, class_id):
        new_label = label
        new_label[new_label != class_id] = 0
        return new_label

if __name__ == '__main__':
    voc2012 = VOC2012('h:/VOC2012/', 'h:/VOC2012/SegmentationClassAug/', image_size=(513, 513))
    voc2012.start_batch_aug_queue(batch_size=8)
    import time
    time.sleep(4)
    for i in range(40):
        start_time = time.time()
        voc2012.get_batch_aug_fast(batch_size=8, random_resize=False)
        print(time.time() - start_time, voc2012.aug_queue.qsize())
