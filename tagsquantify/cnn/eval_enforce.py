# coding:utf-8
from PIL import Image
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import NUS_net_enforce

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/home/wangxiaopeng/NUS_train_sigmo2',
                           """Directory where to read model checkpoints.""")

IMAGE_SIZE = 80


# 获取该图片对应的输入向量
def getimg(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
    # Subtract off the mean and divide by the variance of the pixels.
    var = np.var(re_img)
    return np.divide(np.subtract(re_img, np.mean(re_img)), var)

def evaluate():
    with tf.Graph().as_default() as g:
        num = 2
        arr = tf.placeholder("float", [num, 80, 80, 3])
        logits = NUS_net_enforce.inference(arr,num)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)

            else:
                print('No checkpoint file found')
                return

            print('hehiehiehie..............')

            logit = sess.run(logits[0]-logits[1], feed_dict={#100011951.jpg
                arr: [getimg('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341/100011951.jpg'),
                      getimg('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341/2284611259.jpg')]})
            # logit = sess.run(logits , feed_dict={  # 100011951.jpg
            #     arr: [getimg('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341/100011951.jpg') ]})
            print(logit)
            print(np.array(logit).shape)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
