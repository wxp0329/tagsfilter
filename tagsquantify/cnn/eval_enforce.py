# coding:utf-8
from PIL import Image
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
# import NUS_layers
from three_pics_pairs import  Three_net_enforce
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/wangxiaopeng/Three_train_dir',
                           """Directory where to read model checkpoints.""")

IMAGE_SIZE = 60


# 获取该图片对应的输入向量
def getimg(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    # Subtract off the mean and divide by the variance of the pixels.
    var = np.std(re_img)
    return np.divide(np.subtract(re_img, np.mean(re_img)), var)




def evaluate():
    with tf.Graph().as_default() as g:
        num =2
        arr = tf.placeholder("float", [None, IMAGE_SIZE, IMAGE_SIZE, 3])
        logits = Three_net_enforce.inference(arr,num)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

            else:
                print('No checkpoint file found')
                return

            print('hehiehiehie..............')

            logit = sess.run(tf.reduce_sum(tf.square(logits[0]-logits[1])), feed_dict={#2542737901.jpg
                arr: [getimg('/home/wangxiaopeng/NUS_dataset/images/100011951.jpg'),
                      getimg('/home/wangxiaopeng/NUS_dataset/images/2542737901.jpg')]})
            # logit = sess.run(logits , feed_dict={  # 100011951.jpg
            #     arr: [getimg('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341/2542737901.jpg') ]})
            print(logit)
            print(np.sum(logit))
            print(np.array(logit).shape)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
