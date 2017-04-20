# encoding=utf-8

import re
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 258,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the NUS data set.
LOSS_LAMBDA = 1.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 220000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 300  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1e-3  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 1e-3  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
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
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)

    tf.summary.histogram(tensor_name + '/activations', x)

    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


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
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference1(images, batch=FLAGS.batch_size):
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
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64],
                                             stddev=0.01, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

        # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64],
                                             stddev=0.0589, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

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
                                              stddev=0.0589, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.072, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    drop = tf.nn.dropout(local4, keep_prob=0.5)
    # # # # affine
    with tf.variable_scope('affine') as scope:
        weights = _variable_with_weight_decay('weights', [192, 48],
                                              stddev=0.01, wd=0.0)
        biases = _variable_on_cpu('biases', [48],
                                  tf.constant_initializer(0.))
        affine = tf.nn.relu(tf.matmul(drop, weights) + biases, name=scope.name)
        _activation_summary(affine)

    return affine


def inference(images, batch=FLAGS.batch_size):
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
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64],
                                             stddev=0.272, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv1)

        # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64],
                                             stddev=0.0589, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)

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
                                              stddev=0.0589, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.))
        local3 = tf.nn.sigmoid(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.072, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.))
        local4 = tf.nn.sigmoid(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    drop = tf.nn.dropout(local4, keep_prob=0.5)
    # # # # affine
    with tf.variable_scope('affine') as scope:
        weights = _variable_with_weight_decay('weights', [192, 128],
                                              stddev=0.1, wd=0.0)
        biases = _variable_on_cpu('biases', [128],
                                  tf.constant_initializer(0.))
        affine = tf.nn.sigmoid(tf.matmul(drop, weights) + biases, name=scope.name)
        _activation_summary(affine)

    return affine


def inference12(images, batch=FLAGS.batch_size):
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
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv1)
    with tf.variable_scope('conv1_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv1)
        # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    drop1 = tf.nn.dropout(pool1, keep_prob=0.5)
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(drop1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)
    with tf.variable_scope('conv2_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    drop2 = tf.nn.dropout(pool2, keep_prob=0.5)
    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(drop2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)
    with tf.variable_scope('conv3_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)

        # pool2
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    drop3 = tf.nn.dropout(pool3, keep_prob=0.5)
    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(drop3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)
    with tf.variable_scope('conv4_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)
    with tf.variable_scope('conv4_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)

        # pool4
    pool4 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    drop4 = tf.nn.dropout(pool4, keep_prob=0.5)

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(drop4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)
    with tf.variable_scope('conv5_2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)
    with tf.variable_scope('conv5_3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512],
                                             stddev=0.1, wd=0.0)
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.sigmoid(bias, name=scope.name)
        _activation_summary(conv2)

        # pool4
    pool5 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    drop5 = tf.nn.dropout(pool5, keep_prob=0.5)
    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in drop5.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(drop5, [batch, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 4096],
                                              stddev=0.1, wd=0.0)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.))
        local3 = tf.nn.sigmoid(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 4096],
                                              stddev=0.072, wd=0.0)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.))
        local4 = tf.nn.sigmoid(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    drop = tf.nn.dropout(local4, keep_prob=0.5)
    # # # # affine
    with tf.variable_scope('affine') as scope:
        weights = _variable_with_weight_decay('weights', [4096, 64],
                                              stddev=0.1, wd=0.0)
        biases = _variable_on_cpu('biases', [64],
                                  tf.constant_initializer(0.))
        affine = tf.nn.sigmoid(tf.matmul(drop, weights) + biases, name=scope.name)
        _activation_summary(affine)

    return affine


def loss(imgs):
    """

    :param true_files: 包含tensors的一维数组
    :param false_files: 包含tensors的一维数组
    :return:
    """
    with tf.variable_scope('loss') as scope:
        print('computer losss........................')
        i_s = imgs[:FLAGS.batch_size / 3]  # 前三分之一是i
        j_s = imgs[FLAGS.batch_size / 3:FLAGS.batch_size * 2 / 3]  # 中间三分之一是j
        k_s = imgs[FLAGS.batch_size * 2 / 3:]  # 最后三分之一是k

        i_k_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(i_s, k_s)), axis=1))  # 一列
        i_j_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(i_s, j_s)), axis=1))  # 一列

        loss = tf.reduce_sum(tf.maximum(0., tf.add(tf.subtract(LOSS_LAMBDA, i_k_dist), i_j_dist)))
    tf.add_to_collection('losses', loss)

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
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    #
    # # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
    # return opt
