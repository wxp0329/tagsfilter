# coding:utf-8
from PIL import Image

import datetime
import numpy as np
import shutil
import tensorflow as tf
import os
from multiprocessing import cpu_count

import threadpool

from three_pics_pairs import Three_net_enforce

# from tagsquantify.cnn import NUS_layers

IMAGE_SIZE = 60
IMAGE_DIR = '/media/wangxiaopeng/maxdisk/NUS_dataset/images_220841'
IMAGE_DIR_500 = '/media/wangxiaopeng/maxdisk/NUS_dataset/images_220841'
CHECKPOINT_DIR = '/home/wangxiaopeng/Three_train_sigmoid'
SAVE_MAT_DIR = '/home/wangxiaopeng/NUS_dataset/enforce_mats'
SAVE_MAT_DIR_500 = '/home/wangxiaopeng/NUS_dataset/enforce_mats_500'
COM_DIR = '/home/wangxiaopeng/NUS_dataset/com_dir/'


def getimg(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    # Subtract off the mean and divide by the variance of the pixels.
    var = np.std(re_img)
    return np.divide(np.subtract(re_img, np.mean(re_img)), var)


    # 整合所以图片对应的输出向量到一个文件中
    # （为了得到该文件夹中文件的个数）


# 把所有原始图片通过训练好的模型映射为固定长度的向量


def get_pic_input2output(file_paths, left, right, path=SAVE_MAT_DIR, img_dir=IMAGE_DIR, batch_size=100):
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            # 读取生产的顺序文件（保证最后的向量顺序与该文件里的文件名顺序相同）

            all_pics = []
            for i in file_paths[left:right]:
                name = os.path.join(img_dir, str(i).strip() + '.jpg')
                all_pics.append(getimg(name))

            # 调用模型部分………………………………………………………………………………………………
            arr = tf.placeholder("float", [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])

            logits = Three_net_enforce.inference(arr, batch_size)
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
            # 保存hash过后的结果
            np.save(os.path.join(path, str(left) + '_mat'), affine)
            print 'save affine: ', left
            return affine


def trans_500():
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_test_picName.txt') as fr:
        file_paths = []
        for i in fr.readlines():
            file_paths.append(i.strip().split(' ')[0])
    affine = get_pic_input2output(file_paths, 0, len(file_paths), path=SAVE_MAT_DIR_500, img_dir=IMAGE_DIR_500,
                                  batch_size=len(file_paths))
    np.save(os.path.join(SAVE_MAT_DIR_500, str(2003) + '_hash_mat_48a'), np.where(affine>=0.5,1,0))
    np.save(os.path.join(SAVE_MAT_DIR_500, str(2003) + '_mat_48a'),affine)

def trans_parts():
    if os.path.exists(SAVE_MAT_DIR):
        shutil.rmtree(SAVE_MAT_DIR)

    os.mkdir(SAVE_MAT_DIR)
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_key_list.txt') as fr:
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

        if i + 100 >= 218738:
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
    np.save(os.path.join(os.path.join(SAVE_MAT_DIR_500, 'combine_pic_mat_48a')),
            np.concatenate([a, b, c]))
    np.save(os.path.join(os.path.join(SAVE_MAT_DIR_500, 'combine_hash_pic_mat_48a')),
            np.where(np.concatenate([a, b, c]) >= 0.5, 1, 0))


if __name__ == '__main__':
    start = datetime.datetime.now()
    trans_parts()
    com_parts()
    com_two()
    trans_500()
    # print np.shape(np.load(('/home/wangxiaopeng/combine_pic.mat.npy')))
    end = datetime.datetime.now()
    print 'consume time is :', (end - start).seconds / 60, 'minutes', (end - start).seconds % 60, ' seconds....'
