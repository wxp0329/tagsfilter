#coding:utf-8
import tensorflow as tf
import numpy as np
import os
#
file_paths = []
for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/10000_images'):
    for file in files:
        # notice: read this file shoud strip '\n'
        file_paths.append(os.path.join('/media/wangxiaopeng/maxdisk/NUS_dataset/10000_images',file)+ '\n')
print len(file_paths)
for i in file_paths:
    try:
        file_contents = tf.read_file(str(i).strip())
        image = tf.image.decode_jpeg(file_contents, 3)
    except Exception,InvalidArgumentError:
        os.remove(i)
        print 'delete :',i.rsplit('/',1)[1]
        continue

# file_contents = tf.read_file(str('').strip())
# image = tf.image.decode_jpeg(file_contents, 3)