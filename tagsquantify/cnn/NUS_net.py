# encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from tagsquantify.cnn import NUS_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('img_label_file', '/home/wxp/NSU_dataset/256_img_label_file.dat',
                           """Path to the NUS data directory.""")
tf.app.flags.DEFINE_string('img_label_pair', '/home/wxp/NSU_dataset/256_pairs_labels.dat',
                           """Path to the NUS data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the NUS data set.
LOSS_LAMBDA = 1
IMAGE_SIZE = NUS_input.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = NUS_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = NUS_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


# def _activation_summary(x):
#     """Helper to create summaries for activations.
#
#     Creates a summary that provides a histogram of activations.
#     Creates a summary that measures the sparsity of activations.
#
#     Args:
#       x: Tensor
#     Returns:
#       nothing
#     """
#     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#     # session. This helps the clarity of presentation on tensorboard.
#     tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
#     tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
#     tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity',
#                                          tf.nn.zero_fraction(x))


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
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
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
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inputs(eval_data):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.img_label_file
    images, labels = NUS_input.inputs(eval_data=eval_data,
                                      img_label_file=data_dir,
                                      batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
    return images, labels


def inference(images):
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
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        # _activation_summary(local3)

    # local4 :[FLAGS.batch_size,192]
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        # _activation_summary(local4)

    return local4  # local4 :[FLAGS.batch_size,192]


# 文件格式：lable1 label2 flag
def read_labels():
    all_pair_labels = []
    with open(FLAGS.img_label_pair) as fr:
        for line in fr.readlines():
            elements = line.strip().split(' ')
            all_pair_labels.append(elements)
    return all_pair_labels


# 判断label_pair 属于那个类
def which_room(labels, i, j):
    print('which_room')
    pair_label = {labels[i], labels[j]}
    for i in read_labels():
        if len(pair_label.intersection(set(i))) == 2:
            return i[2]  # 1 or 0


def loss(imgs, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      imgs: imgs from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels_size = int(str(labels.get_shape()).strip('()')[:-1])
    one_labels = []
    zero_labels = []
    # for i in xrange(labels_size - 1):
    #     for j in xrange(i + 1, labels_size):
    #         pair_sub = tf.sub(imgs[i], imgs[j])
    #         pair_square = tf.square(pair_sub)
    #         pair_sum = tf.reduce_sum(pair_square)
    #         if which_room(labels, i, j):
    #             print ('fang ru 1 ...........',len(one_labels))
    #             one_labels.append(pair_sum)
    #         else:
    #             print('fang ru 0 ...........',len(zero_labels))
    #             # max_sum = np.maximum(tf.constant(0, dtype=tf.float32) , tf.constant(LOSS_LAMBDA, dtype=tf.float32) - pair_sum)
    #             zero_labels.append(tf.cond(tf.greater(tf.constant(0, dtype=tf.float32) , tf.constant(LOSS_LAMBDA, dtype=tf.float32) - pair_sum),
    #                                        lambda:tf.constant(0, dtype=tf.float32), lambda:tf.constant(LOSS_LAMBDA, dtype=tf.float32) - pair_sum))
    for i in xrange(labels_size - 1):
        for j in xrange(i + 1, labels_size):
            pair_sub = tf.sub(imgs[i], imgs[j])
            pair_square = tf.square(pair_sub)
            pair_sum = tf.reduce_sum(pair_square)
            if which_room(labels, i, j):
                print ('fang ru 1 ...........',len(one_labels))
                one_labels.append(pair_sum)
            else:
                print('fang ru 0 ...........',len(zero_labels))
                # max_sum = np.maximum(tf.constant(0, dtype=tf.float32) , tf.constant(LOSS_LAMBDA, dtype=tf.float32) - pair_sum)
                zero_labels.append(tf.cond(tf.greater(tf.constant(0, dtype=tf.float32) , tf.constant(LOSS_LAMBDA, dtype=tf.float32) - pair_sum),
                                           lambda:tf.constant(0, dtype=tf.float32), lambda:tf.constant(LOSS_LAMBDA, dtype=tf.float32) - pair_sum))

    loss_my = tf.add(tf.reduce_sum(ops.convert_to_tensor(one_labels)), tf.reduce_sum(ops.convert_to_tensor(zero_labels)))

    print('loss_my: ',loss_my)

    tf.add_to_collection('losses', loss_my)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """Train CIFAR-10 model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    #
    # # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.scalar(var.op.name, var)

    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         tf.contrib.deprecated.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
    # return opt
