# coding:utf-8
from PIL import Image

import numpy as np
import shutil
import tensorflow as tf
import os
# import NUS_layers
from three_pics_pairs import  Three_net_enforce

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/wangxiaopeng/Three_train_dir',
                           """Directory where to read model checkpoints.""")
IMG_SIZE = 60


# 获取该图片对应的输入向量
def getimg(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    # Subtract off the mean and divide by the variance of the pixels.
    var = np.std(re_img)
    return np.divide(np.subtract(re_img, np.mean(re_img)), var)


# all 100 dimention points of  pics to save file
def generatePoints(arr):
    # loss = NUS_net.loss(logits, tf.Variable([5,7],dtype=tf.int32))    1055282  1055277 1055406

    x = tf.placeholder('float', [1, IMG_SIZE, IMG_SIZE, 3])
    logits = Three_net_enforce.inference(x, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:

            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            logit = sess.run(logits, feed_dict={x: arr})

            # print(np.array(logit).shape)
            return logit
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def get_top_k_indexes(filename, num):
    """filename: the pic name you want to retrive
     num: the matched pics number you want to return"""
    all_pics_datas = np.load('/home/wangxiaopeng/combine_pic.mat.npy')

    oushi_dist = np.sum(np.square(np.subtract(all_pics_datas, filename)), axis=1)

    smallTobig = np.argsort(oushi_dist)
    return smallTobig[0:num]


# according the order of the [210843,100] matrix to sort the img:label pairs of the replaced_All_tags.dat
def write_img_label(sortedFilename, matrix_filename, path2='/home/wangxiaopeng/NUS_dataset/replaced_All_tags.dat'):
    dic = dict()
    with open(path2) as fr:
        for line in fr.readlines():
            d = line.strip().split('__#__')
            dic[d[0]] = d[1]
    print 'replaced_All_tags.dat: ', len(dic)

    matrix_filenames = []
    with open(matrix_filename) as fr:
        for line in fr.readlines():
            # pic name without suffix
            matrix_filenames.append(line.strip().rsplit('/', 1)[1].split('.')[0])

    print 'matrix_filenames :', len(matrix_filenames)
    with open(sortedFilename, 'a') as fw:
        for name in matrix_filenames:
            fw.write(name + '__^__' + dic[name])


#
def getsortedFilenames(path):
    """get the sortedFilenames (img:labels) pairs from write_img_label()
        img: the key
        labels: the set
        return: dict"""
    named_dic = dict()
    num_dic = dict()
    i = 0
    with open(path) as fr:
        for line in fr.readlines():
            line = line.strip().split('__^__')
            s = set()
            for item in line[1]:
                s.add(item.strip())
            named_dic[line[0]] = s
            num_dic[line[0]] = i
            i += 1
    return named_dic, num_dic


# 通过用户输入一个图片，返回给图片对应的top_n个图片
def get_one2more(user_pic_name, found_files_dir='/home/wangxiaopeng/found_pics/',
                 all_pics_name_file='/media/wangxiaopeng/maxdisk/NUS_dataset/220341_pics_names.txt'
                 ):
    with tf.Graph().as_default() as g:
        with open(all_pics_name_file) as fr:
            all_file_name = fr.readlines()
            print 'creating :', user_pic_name, ' similar pics.....'
        original = user_pic_name
        # before finding remove all previus pics
        # removeAllPics(found_files_dir)
        os.mkdir(found_files_dir)
        with open(original) as fr:
            pic = fr.readlines()
            with open(found_files_dir + '0_original_' + original.strip().rsplit('/', 1)[1], 'w') as fw:
                fw.writelines(pic)

        original = generatePoints([getimg(original)])
        num = 1
        for i in get_top_k_indexes(original, 10):
            name = all_file_name[i].strip()
            # print i, name
            with open(os.path.join('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341/', name)) as fr:
                pic = fr.readlines()
                with open(found_files_dir + str(num) + '_' + str(name), 'w') as fw:
                    fw.writelines(pic)
            num += 1


def main(argv=None):  # pylint: disable=unused-argument

    # get_pic_input2output()
    file_paths = []
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/500_img_names.txt') as fr:

        for i in fr.readlines():
            file_paths.append(os.path.join('/media/wangxiaopeng/maxdisk/NUS_dataset/images_500/', i.strip()))

    if os.path.exists('/home/wangxiaopeng/found_pics/'):
        shutil.rmtree('/home/wangxiaopeng/found_pics/')

    os.mkdir('/home/wangxiaopeng/found_pics/')

    num = 1
    for i in file_paths[:100]:
        get_one2more(i, '/home/wangxiaopeng/found_pics/' + str(num) + '_sample/')
        num += 1
        # com()


if __name__ == '__main__':
    tf.app.run()
