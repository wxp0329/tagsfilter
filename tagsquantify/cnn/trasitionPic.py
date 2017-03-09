# coding:utf-8
from PIL import Image

import datetime
import numpy as np
import shutil
import tensorflow as tf
import os
from multiprocessing import cpu_count

import threadpool

# from tagsquantify.cnn import NUS_net_enforce

IMAGE_SIZE = 80
IMAGE_DIR = '/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341'
CHECKPOINT_DIR = '/home/wangxiaopeng/NUS_train_sigmo2'
SAVE_MAT_DIR = '/home/wangxiaopeng/NUS_dataset/enforce_mats'
COM_DIR = '/home/wangxiaopeng/NUS_dataset/com_dir/'


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(images, batch=100):
    """Build the NUS_dataset model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=2e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

        # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=2e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [batch, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # # # # affine
    with tf.variable_scope('NUS_affine') as scope:
        weights = _variable_with_weight_decay('weights', [192, 10],
                                              stddev=1 / 100.0, wd=0.0)
        biases = _variable_on_cpu('biases', [10],
                                  tf.constant_initializer(0.0))
        affine = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)

    return affine


# 获取该图片对应的输入向量
def getimg(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    # Subtract off the mean and divide by the variance of the pixels.
    var = np.std(re_img)
    return np.divide(np.subtract(re_img, np.mean(re_img)), var)


    # 整合所以图片对应的输出向量到一个文件中
    # （为了得到该文件夹中文件的个数）


# 把所有原始图片通过训练好的模型映射为固定长度的向量


def get_pic_input2output(file_paths, left, right):
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            # 读取生产的顺序文件（保证最后的向量顺序与该文件里的文件名顺序相同）

            all_pics = []
            for i in file_paths[left:right]:
                name = os.path.join(IMAGE_DIR, str(i).strip())
                all_pics.append(getimg(name))

            # 调用模型部分………………………………………………………………………………………………
            arr = tf.placeholder("float", [100, IMAGE_SIZE, IMAGE_SIZE, 3])

            logits = inference(arr, 100)
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
            # 读取所有图片的.npy文件的个数（为了得到该文件夹中文件的个数）

            affine = sess.run(logits, feed_dict={
                arr: all_pics})

            np.save(os.path.join(SAVE_MAT_DIR, str(left) + '_mat'), affine)
            print 'save affine: ', left


def trans_parts():
    if os.path.exists(SAVE_MAT_DIR):
        shutil.rmtree(SAVE_MAT_DIR)

    os.mkdir(SAVE_MAT_DIR)
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/220341_pics_names.txt') as fr:
        file_paths = fr.readlines()
    print 'cpu_count :', cpu_count()
    len_ = len(file_paths)
    i_list = []
    for i in xrange(0, 230000, 100):  # 3000000 行大约50M数据

        if i + 100 >= len_:
            i_list.append([file_paths, i, len_ + 1])
            break
        i_list.append([file_paths, i, i + 100])
    n_list = [None for i in range(len(i_list))]
    pool = threadpool.ThreadPool(cpu_count())
    requests = threadpool.makeRequests(get_pic_input2output, zip(i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print 'all of strans_part excute over !!!!!'


def com(file_id, left, right):
    all_mat_after = []
    for i in file_id[left:right]:
        all_mat_after.append(
            np.load(os.path.join(SAVE_MAT_DIR, str(i) + '_mat.npy')))

    np.save(os.path.join(COM_DIR, str(left) + '_combine_pic.mat'),
            np.concatenate(all_mat_after))
    print 'combination over!!!'


def com_parts():
    if os.path.exists(COM_DIR):
        shutil.rmtree(COM_DIR)

    os.mkdir(COM_DIR)

    print 'cpu_count :', cpu_count()
    file_id = []
    for i in xrange(0, 230000, 100):  # generate files indexes

        if i + 100 >= 220241:
            file_id.append(i)
            break
        file_id.append(i)

    len_ = len(file_id)

    com_i_list = []
    for i in xrange(0, 230000, 1000):  # split files indexes

        if i + 1000 >= len_:
            com_i_list.append([file_id, i, len_])
            break
        com_i_list.append([file_id, i, i + 1000])
    n_list = [None for ii in range(len(com_i_list))]
    pool = threadpool.ThreadPool(cpu_count())
    requests = threadpool.makeRequests(com, zip(com_i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print 'all of com excute over !!!!!'


def com_two():
    a = np.load(os.path.join('/home/wangxiaopeng/NUS_dataset/com_dir/0_combine_pic.mat.npy'))
    b = np.load(os.path.join('/home/wangxiaopeng/NUS_dataset/com_dir/1000_combine_pic.mat.npy'))
    c = np.load(os.path.join('/home/wangxiaopeng/NUS_dataset/com_dir/2000_combine_pic.mat.npy'))
    np.save(os.path.join('/home/wangxiaopeng/combine_pic.mat'),
            np.concatenate([a, b, c]))


if __name__ == '__main__':
    start = datetime.datetime.now()
    trans_parts()
    # com_parts()
    # com_two()
    # print np.shape(np.load(('/home/wangxiaopeng/combine_pic.mat.npy')))
    end = datetime.datetime.now()
    print 'consume time is :',(end-start).seconds,' seconds....'
