# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('img_size', 60,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('step_skip', 1000,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('imgs_dir',
                           '/home/wangxiaopeng/NUS_dataset/images_220341',
                           """Path to the NUS data directory.""")
tf.app.flags.DEFINE_string('true_files_dir',
                           '/home/wangxiaopeng/NUS_dataset/true_files',
                           """每行代表索引对应与220341_key_list.dat的pairs""")
tf.app.flags.DEFINE_string('false_files_dir',
                           '/home/wangxiaopeng/NUS_dataset/false_files',
                           """每行代表索引对应与220341_key_list.dat的pairs""")
tf.app.flags.DEFINE_string('mid_files_dir',
                           '/home/wangxiaopeng/NUS_dataset/mid_files',
                           """交集为0的pairs. ,每行代表文件名的pairs ，与其他true_files和true_files不同""")
tf.app.flags.DEFINE_string('filenames_list_file',
                           '/home/wangxiaopeng/NUS_dataset/220341_key_list.dat',
                           """Path to the NUS data directory.""")


class InputUtil:
    def __init__(self, imgs_dir_name):
        self.IMG_SIZE = 80
        self.imgs_dir_name = imgs_dir_name
        # 读取文件名列表
        with open(FLAGS.filenames_list_file) as fr:
            self.paths = fr.readlines()

        # 获取正例文件列表
        true_files = []
        for root, dirs, files in os.walk(FLAGS.true_files_dir):
            for file in files:
                # notice: read this file shoud strip '\n'
                with open(os.path.join(root, file)) as fr:
                    true_files = (fr.readlines())
        true_indexes = []
        true_len = len(true_files)
        for i in xrange(1000, true_len, 1000):
            true_indexes.append(i)

        self.true_files = np.split(np.array(true_files), true_indexes)
        self.true_files_len = len(self.true_files)

        print 'read true_files over !!!'
        # 获取负例文件列表
        false_files = []
        for root, dirs, files in os.walk(FLAGS.false_files_dir):
            for file in files:
                # notice: read this file shoud strip '\n'
                with open(os.path.join(root, file)) as fr:
                    false_files = (fr.readlines())

        false_indexes = []
        false_len = len(false_files)
        for i in xrange(1000, false_len, 1000):
            false_indexes.append(i)
        self.false_files = np.split(np.array(false_files), false_indexes)
        self.false_files_len = len(self.false_files)
        print 'read false_files over !!!'

        mid_files = []
        for root, dirs, files in os.walk(FLAGS.mid_files_dir):
            for file in files:
                # notice: read this file shoud strip '\n'
                with open(os.path.join(root, file)) as fr:
                    mid_files = (fr.readlines())
        mid_indexes = []
        mid_len = len(mid_files)
        for i in xrange(1000, mid_len, 1000):
            mid_indexes.append(i)
        self.mid_files = np.split(np.array(mid_files), mid_indexes)
        self.mid_files_len = len(self.mid_files)
        print 'read mid_files over !!!'

    # 获取该图片对应的输入向量
    def getimg(self, str1):

        im = Image.open(str1)
        re_img = np.array(im.resize((FLAGS.img_size, FLAGS.img_size)), dtype=np.float32)
        # Subtract off the mean and divide by the variance of the pixels.
        std = np.std(re_img)
        return np.divide(np.subtract(re_img, np.mean(re_img)), std)

    def read_true_sample(self, batch_size, step):
        # 随机读取false_files中的一个文件（1000行数据），再随机返回batch_size / 8个pairs
        part_true_names = np.random.choice(self.true_files[step%self.true_files_len], batch_size / 4,
                                           replace=False)
        # 获取打乱后的正例文件名
        true_pair_lefts = []  # 存放的每个pair的左部文件集
        true_pair_rights = []  # 存放的每个pair的右部文件集
        for i in part_true_names:
            pair = i.strip().split(' ')
            true_pair_lefts.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[0])].strip() + '.jpg')))
            true_pair_rights.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[1])].strip() + '.jpg')))
        # 一维数组
        return [true_pair_lefts, true_pair_rights]

    def read_false_sample(self, batch_size, step):
        # 随机读取false_files中的一个文件（1000行数据），再随机返回batch_size / 8个pairs
        part_false_names = np.random.choice(self.false_files[step%self.false_files_len],
                                            batch_size / 8, replace=False)
        part_mid_names = np.random.choice(self.mid_files[step%self.mid_files_len], batch_size / 8,
                                          replace=False)
        # 获取打乱后的正例文件名
        false_pair_lefts = []  # 存放的每个pair的左部文件集
        false_pair_rights = []  # 存放的每个pair的右部文件集
        for i in part_false_names:
            pair = i.strip().split(' ')
            false_pair_lefts.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[0])].strip() + '.jpg')))
            false_pair_rights.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[1])].strip() + '.jpg')))

        # 获取打乱后的正例文件名
        mid_pair_lefts = []  # 存放的每个pair的左部文件集
        mid_pair_rights = []  # 存放的每个pair的右部文件集
        for i in part_mid_names:
            pair = i.strip().split(' ')  # 注意： 这个pairs就是实际的文件名！！！！！！！！！
            mid_pair_lefts.append(self.getimg(os.path.join(self.imgs_dir_name, pair[0].strip() + '.jpg')))
            mid_pair_rights.append(self.getimg(os.path.join(self.imgs_dir_name, pair[1].strip() + '.jpg')))
        return [np.concatenate([false_pair_lefts, mid_pair_lefts]),
                np.concatenate([false_pair_rights, mid_pair_rights])]

    # 返回下一批数据
    def next_batch(self, batch_size, step):
        # 数据格式[batch,weight,high,chanel]
        # with tf.variable_scope('batch_input') as scope:
        true_pairs = self.read_true_sample(batch_size, step)
        false_mid_pairs = self.read_false_sample(batch_size, step)
        datas = np.concatenate([true_pairs[0], false_mid_pairs[0], true_pairs[1], false_mid_pairs[1]])
        return datas
