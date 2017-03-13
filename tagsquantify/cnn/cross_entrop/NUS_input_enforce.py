# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('img_size', 60,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('imgs_dir',
                           '/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341',
                           """Path to the NUS data directory.""")
tf.app.flags.DEFINE_string('name_index_file',
                           '',
                           """每行代表索引对应与220341_key_list.dat的pairs""")

tf.app.flags.DEFINE_string('all_tags_file',
                           '/media/wangxiaopeng/maxdisk/NUS_dataset/tags/AllTags81.txt',
                           """每行代表索引对应与220341_key_list.dat的pairs""")

class InputUtil:

    def __init__(self, imgs_dir_name):
        self.imgs_dir_name = imgs_dir_name
        self.all_tags = np.loadtxt(FLAGS.all_tags_file,dtype=np.string_)
        self.name_indexes = np.loadtxt(FLAGS.all_tags_file,dtype=np.string_)


    # 获取该图片对应的输入向量
    def getimg(self, str1):
        im = Image.open(str1)
        re_img = np.array(im.resize((FLAGS.img_size, FLAGS.img_size)), dtype=np.float32)
        # Subtract off the mean and divide by the variance of the pixels.
        std = np.std(re_img)
        return np.divide(np.subtract(re_img, np.mean(re_img)), std)

    def next_batch(self, batch_size):
        np.random.shuffle(self.name_indexes)
        pics = []
        name_labels = self.name_indexes[batch_size]
        for pic in name_labels[:,0]:
            pics.append(self.getimg(os.path.join(self.imgs_dir_name,pic+'.jpg')))

        labels = self.all_tags[np.array(name_labels[:,1],dtype=np.int)]
        return pics,labels

