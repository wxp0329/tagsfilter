from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math
import tensorflow as tf

from tagsquantify.cnn import NUS_input
from tagsquantify.cnn import NUS_net
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/wxp/NUS_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# Get images and labels for CIFAR-10.
images, labels = NUS_input.inputs(False, '/home/wxp/NSU_dataset/img_index_1.dat',NUS_net.FLAGS.batch_size)

# Build a Graph that computes the logits predictions from the
# inference model.
print ('hahh.....^^^^^^^^^^^^')
print (tf.shape(images))
print (tf.shape(labels))
logits = NUS_net.inference(images)

# Calculate loss.
loss = NUS_net.loss(logits, labels)
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/home/wxp/NUS_train')
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ('hhahaahahh!!!!!!!!!!!!!!!')
        print (sess.run(loss))
        print(sess.run(logits))
