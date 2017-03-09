# coding:utf-8
from PIL import Image

import numpy as np
import shutil
import tensorflow as tf
import os

import NUS_net_enforce

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/wangxiaopeng/NUS_train_sigmo2',
                           """Directory where to read model checkpoints.""")
ALL_POINTS_DIR = '/media/wangxiaopeng/maxdisk/NUS_dataset/'
IMG_SIZE = 80


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

    x = tf.placeholder('float', [1, IMG_SIZE,IMG_SIZE,3])
    logits = NUS_net_enforce.inference(x, 1)
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
            logit = sess.run(logits,feed_dict={x:arr})

            print(np.array(logit).shape)
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
    return smallTobig[1:num]


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


# 把所有原始图片通过训练好的模型映射为固定长度的向量
def get_pic_input2output(after_mat_dir='/home/wangxiaopeng/NUS_dataset/5000_mats_after/',
                         before_mat_dir='/home/wangxiaopeng/NUS_dataset/5000_mats_before/',
                         imgs_dir='/media/wangxiaopeng/maxdisk/NUS_dataset/images_210841',
                         all_name_file='/media/wangxiaopeng/maxdisk/NUS_dataset/220342_pics_names.txt'):
    with tf.Graph().as_default() as g:
        img = tf.placeholder(dtype=tf.float32,shape=[None,60,60,3])
        with tf.Session() as sess:
            # 读取生产的顺序文件（保证最后的向量顺序与该文件里的文件名顺序相同）
            with open(all_name_file) as fr:
                file_paths = fr.readlines()[74500:100000]

            num = 745
            m = 0
            # 把每个图片对应的输入向量保存在10000_mats文件夹中
            while True:
                n = m + 100
                if n >= len(file_paths):
                    all_pics = []
                    for i in file_paths[m:n]:
                        name = os.path.join(imgs_dir, str(i).strip())
                        all_pics.append(sess.run(img,feed_dict={img:[getimg(name)]}))

                    np.save(before_mat_dir + str(num) + '_mat', all_pics)
                    print 'save:', num
                    break
                all_pics = []
                for i in file_paths[m:n]:
                    name = os.path.join(imgs_dir, str(i).strip())
                    all_pics.append(sess.run(img, feed_dict={img: [getimg(name)]}))

                np.save(before_mat_dir + str(num) + '_mat', all_pics)
                print 'save:', num
                num += 1
                m = n

            # 调用模型部分………………………………………………………………………………………………
            arr = tf.placeholder("float", [1000, 60, 60, 3])
            logits = NUS_net_enforce.inference(arr, 1000)
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
            # 读取所有图片的.npy文件的个数（为了得到该文件夹中文件的个数）
            file_paths = []
            for root, dirs, files in os.walk(before_mat_dir):
                for file in files:
                    # notice: read this file shoud strip '\n'
                    file_paths.append(file + '\n')
            #
            # 把各个输入图片的向量对应的输出向量保存到10000_mats_after
            for i in range(300):
                affine = sess.run(logits, feed_dict={
                    arr: np.load(before_mat_dir + str(i) + '_mat.npy')})

                np.save(after_mat_dir + str(i) + '_mat', affine)
                print 'save affine: ', i

        # 整合所以图片对应的输出向量到一个文件中
        # （为了得到该文件夹中文件的个数）
        file_paths = []
        for root, dirs, files in os.walk(after_mat_dir):
            for file in files:
                # notice: read this file shoud strip '\n'
                file_paths.append(file + '\n')

        all_mat_after = []
        for i in range(len(file_paths)):
            all_mat_after.append(
                np.load(after_mat_dir + str(i) + '_mat.npy'))

        np.save(os.path.join(after_mat_dir, 'combine_pic.mat'),
                np.concatenate(all_mat_after))
        print 'combination over!!!'


# 在生成新的图片集时，删除之前存在的图片
def removeAllPics(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # notice: read this file shoud strip '\n'
            file_paths.append(os.path.join(root, file) + '\n')
    print len(file_paths)
    for i in file_paths:
        os.remove(i.strip())


# 通过用户输入一个图片，返回给图片对应的top_n个图片
def get_one2more(user_pic_name, found_files_dir='/home/wangxiaopeng/found_pics/',
                 all_pics_name_file='/media/wangxiaopeng/maxdisk/NUS_dataset/220341_pics_names.txt'
                 ):
    with tf.Graph().as_default() as g:
        with open(all_pics_name_file) as fr:
            all_file_name = fr.readlines()
            print 'lines:', len(all_file_name)
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
        for i in get_top_k_indexes(original, 11):
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
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/images_500'):
        for file in files:
            # notice: read this file shoud strip '\n'
            file_paths.append(os.path.join(root, file))

    if os.path.exists('/home/wangxiaopeng/found_pics/'):
        shutil.rmtree('/home/wangxiaopeng/found_pics/')

    os.mkdir('/home/wangxiaopeng/found_pics/')

    num = 1
    for i in file_paths[:50]:
        get_one2more(i, '/home/wangxiaopeng/found_pics/' + str(num) + '_sample/')
        num += 1
    # com()

if __name__ == '__main__':
    tf.app.run()
