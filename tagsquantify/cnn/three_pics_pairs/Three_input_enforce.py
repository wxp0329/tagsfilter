# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('img_size', 60,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('pair_dir', '/',
                           """three pairs files dir.""")
tf.app.flags.DEFINE_string('imgs_dir',
                           '/home/wangxiaopeng/NUS_dataset/images_220341',
                           """Path to the NUS data directory.""")
tf.app.flags.DEFINE_string('filenames_list_file',
                           '/home/wangxiaopeng/NUS_dataset/220341_key_list.dat',
                           """Path to the NUS data directory.""")


class InputUtil:
    def __init__(self):
        # 读取文件名列表
        with open(FLAGS.filenames_list_file) as fr:
            self.paths = fr.readlines()

        with open(FLAGS.pair_dir) as fr:  # 文件格式：i j k
            pair_files = fr.readlines()

        pair_indexes = []
        pair_len = len(pair_files)
        for i in xrange(1000, pair_len, 1000):
            pair_indexes.append(i)

        self.pair_files = np.split(np.array(pair_files), pair_indexes)  # 1000行为一个块
        self.pair_files_len = len(self.pair_files)

        print 'read pair_files over !!!'

    # 获取该图片对应的输入向量
    def getimg(self, str1):
        im = Image.open(str1)
        re_img = np.array(im.resize((FLAGS.img_size, FLAGS.img_size)), dtype=np.float32)
        # Subtract off the mean and divide by the variance of the pixels.
        std = np.std(re_img)
        return np.divide(np.subtract(re_img, np.mean(re_img)), std)

    def next_batch(self, batch_size, step):
        # 随机读取false_files中的一个文件（1000行数据），再随机返回batch_size / 8个pairs
        part_pair_names = np.random.choice(self.pair_files[step % self.pair_files_len], batch_size / 3,
                                           replace=False)
        i_s = []
        j_s = []
        k_s = []
        for line in part_pair_names:
            line = line.strip().split(' ')
            i_s.append(self.getimg(os.path.join(FLAGS.imgs_dir,self.paths[line[0]]+'.jpg')))
            j_s.append(self.getimg(os.path.join(FLAGS.imgs_dir, self.paths[line[1]] + '.jpg')))
            k_s.append(self.getimg(os.path.join(FLAGS.imgs_dir, self.paths[line[2]] + '.jpg')))
        # 一维数组
        return [np.concatenate([i_s,j_s,k_s])]