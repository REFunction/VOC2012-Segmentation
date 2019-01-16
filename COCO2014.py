import cv2
import numpy as np
import os
import random
import threading
import queue

class COCO2014:
    def __init__(self, image_folder_path, label_folder_path, image_size=(500, 500)):
        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        if image_folder_path[len(image_folder_path) - 1] != '/' \
                and image_folder_path[len(image_folder_path) - 1] != '\\':
            self.image_folder_path += '/'
        if label_folder_path[len(label_folder_path) - 1] != '/' \
                and label_folder_path[len(label_folder_path) - 1] != '\\':
            self.label_folder_path += '/'
        if not os.path.isdir(image_folder_path):
            print('image folder ', image_folder_path, 'does not exist')
        if not os.path.isdir(label_folder_path):
            print('label folder ', label_folder_path, 'does not exist')
        self.image_size = image_size
        self.read_image_names()
        self.read_label_names()
    def read_image_names(self):
        self.image_names = os.listdir(self.image_folder_path)
    def read_label_names(self):
        self.label_names = os.listdir(self.label_folder_path)
    def get_batch(self, batch_size):
        if hasattr(self, 'location') == False:
            self.location = 0
        end = min(self.location + batch_size, len(self.image_names))
        start = self.location
        batch_images_names = self.image_names[start:end]
        batch_labels_names = self.label_names[start:end]
        self.location = (self.location + batch_size) % len(self.image_names)
        if end - start != batch_size:
            batch_images_names = np.concatenate([batch_images_names, self.image_names[0:self.location]], axis=0)
            batch_labels_names = np.concatenate([batch_labels_names, self.label_names[0:self.location]], axis=0)

        batch_images = []
        batch_labels = []
        for i in range(batch_size):
            if batch_images_names[i][:-4] != batch_labels_names[i][:-4]:
                print('Images and labels are inconsisent')
                exit()
            image = cv2.imread(self.image_folder_path + batch_images_names[i])
            image = cv2.resize(image, self.image_size)
            label = cv2.imread(self.label_folder_path + batch_labels_names[i], cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            batch_images.append(image)
            batch_labels.append(label)
        return batch_images, batch_labels
    def add_batch_queue(self, batch_size, max_queue_size):
        if hasattr(self, 'batch_queue') == False:
            self.batch_queue = queue.Queue(maxsize=max_queue_size)
        while 1:
            image_batch, label_batch = self.get_batch(batch_size)
            image_batch, label_batch = self.random_resize(image_batch, label_batch)
            self.batch_queue.put([image_batch, label_batch])
    def start_batch_queue(self, batch_size, max_queue_size=30):
        if hasattr(self, 'batch_queue') == False:
            queue_thread = threading.Thread(target=self.add_batch_queue, args=(batch_size, max_queue_size))
            queue_thread.start()
    def get_batch_fast(self, batch_size, max_queue_size=30):
        '''
        A fast function for get batch.Use another thread to get batch and put into a queue.
        :param batch_size: batch size
        :param max_queue_size: the max capacity of the queue
        :return: An image batch with shape [batch_size, height, width, 3]
                and a label batch with shape [batch_size, height, width, 1]
        '''
        # create queue thread
        if hasattr(self, 'batch_queue') == False:
            queue_thread = threading.Thread(target=self.add_batch_queue, args=(batch_size, max_queue_size))
            queue_thread.start()
        while hasattr(self, 'batch_queue') == False:
            time.sleep(0.1)
        image_batch, label_batch = self.batch_queue.get()
        return image_batch, label_batch

    def random_resize(self, image_batch, label_batch, random_blur=True):
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
