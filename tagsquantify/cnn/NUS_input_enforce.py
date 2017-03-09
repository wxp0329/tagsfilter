# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from PIL import Image

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('img_size', 80,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('step_skip', 1000,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('imgs_dir',
                           '/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341',
                           """Path to the NUS data directory.""")
tf.app.flags.DEFINE_string('true_files_dir',
                           '/home/wangxiaopeng/NUS_dataset/true_files',
                           """每行代表索引对应与220341_key_list.dat的pairs""")
tf.app.flags.DEFINE_string('false_files_dir',
                           '/home/wangxiaopeng/NUS_dataset/true_files',
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
                true_files.append(os.path.join(root, file))
        self.true_files = np.array(true_files)
        # 获取负例文件列表
        false_files = []
        for root, dirs, files in os.walk(FLAGS.false_files_dir):
            for file in files:
                # notice: read this file shoud strip '\n'
                false_files.append(os.path.join(root, file))
        self.false_files = np.array(false_files)
        # 获取交集为0文件列表
        mid_files = []
        for root, dirs, files in os.walk(FLAGS.mid_files_dir):
            for file in files:
                # notice: read this file shoud strip '\n'
                mid_files.append(os.path.join(root, file))
        self.mid_files = np.array(mid_files)

    # 获取该图片对应的输入向量
    def getimg(self, str1):

        im = Image.open(str1)
        re_img = np.array(im.resize((FLAGS.img_size, FLAGS.img_size)), dtype=np.float32)
        # Subtract off the mean and divide by the variance of the pixels.
        std = np.std(re_img)
        return np.divide(np.subtract(re_img, np.mean(re_img)), std)

    def read_true_sample(self, batch_size, step):
        if step == 0:  # 正例就一个文件，只需要读取一次
            np.random.shuffle(self.true_files)
            # 获取打乱后的第一个文件
            with open(self.true_files[0]) as fr:
                self.true_names = np.array(fr.readlines())  # 格式：0 122 1

        np.random.shuffle(self.true_names)
        # 获取打乱后的正例文件名
        true_pair_lefts = []  # 存放的每个pair的左部文件集
        true_pair_rights = []  # 存放的每个pair的右部文件集
        for i in self.true_names[:batch_size / 4]:
            pair = i.strip().split(' ')
            true_pair_lefts.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[0])].strip() + '.jpg')))
            true_pair_rights.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[1])].strip() + '.jpg')))
        # 一维数组
        return [true_pair_lefts, true_pair_rights]

    def read_false_sample(self, batch_size, step):
        if step % FLAGS.step_skip == 0:
            np.random.shuffle(self.false_files)
            np.random.shuffle(self.mid_files)
            # 获取打乱后的第一个文件
            with open(self.false_files[0]) as fr:
                self.false_names = np.array(fr.readlines())
            with open(self.mid_files[0]) as fr1:
                self.mid_names = np.array(fr1.readlines())

        np.random.shuffle(self.false_names)
        # 获取打乱后的正例文件名
        false_pair_lefts = []  # 存放的每个pair的左部文件集
        false_pair_rights = []  # 存放的每个pair的右部文件集
        for i in self.false_names[:batch_size / 8]:
            pair = i.strip().split(' ')
            false_pair_lefts.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[0])].strip() + '.jpg')))
            false_pair_rights.append(
                self.getimg(os.path.join(self.imgs_dir_name, self.paths[int(pair[1])].strip() + '.jpg')))

        np.random.shuffle(self.mid_names)
        # 获取打乱后的正例文件名
        mid_pair_lefts = []  # 存放的每个pair的左部文件集
        mid_pair_rights = []  # 存放的每个pair的右部文件集
        for i in self.mid_names[:batch_size / 8]:
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
