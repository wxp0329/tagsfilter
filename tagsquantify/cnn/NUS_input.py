# encoding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os

IMAGE_SIZE = 100
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
IMG_ROOT_DIR = '/home/wxp/NSU_dataset/images'


def read_labeled_image_list(image_list_file):
    """
    Read a .txt file containing pathes and labeles.
    Parameters
    ----------
     image_list_file : a .txt file with one /path/to/image per line
     label : optionally, if set label will be pasted after each line
    Returns
    -------
       List with all filenames in file image_list_file
    """
    # 图片数据少于标签集中已记录的个数，加上这个操作过滤掉图片集中没有的路径，防止读取图片出错
    file_paths = []
    for root, dirs, files in os.walk(IMG_ROOT_DIR):  # 图片集文件夹中的所有存在的图片路径
        for file in files:
            file_paths.append(file)

    exist_pic_path = set()  # 真实存在的图片的路径（因为有些图片没有下载到！！！）
    f = open(image_list_file, 'r')
    for i in f.readlines():
        pic_name = i.strip().split(' ')[0]
        if pic_name in file_paths:
            exist_pic_path.add(i)

    print('总共参与训练的图片数为：', len(exist_pic_path), len(file_paths))
    f.close()
    filenames = []
    labels = []
    for line in exist_pic_path:
        filename, label = line.strip().split(' ')
        filenames.append(os.path.join(IMG_ROOT_DIR, filename))
        labels.append(label)
    return filenames, labels


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Parameters
    ----------
      filename_and_label_tensor: A scalar string tensor.
    Returns
    -------
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]

    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, 3)
    # example = rescale_image(example)
    # processed_label = label
    return example, label


def inputs(eval_data, img_label_file, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    image_list, label_list = read_labeled_image_list(img_label_file)

    if not eval_data:

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in image_list:
        if not tf.gfile.Exists(os.path.join(IMG_ROOT_DIR, f)):
            raise ValueError('Failed to find file: ' + f)

            # Create a queue that produces the filenames to read.
            # filename_queue = tf.train.string_input_producer(filenames)

            # Read examples from files in the filename queue.
            # reader = tf.WholeFileReader()
            # key, value = reader.read(filename_queue)
            # Reads pfathes of images together with there labels

    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    # labels = ops.convert_to_tensor(label_list, dtype=dtypes.string)
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, label_list],
                                                num_epochs=num_examples_per_epoch,
                                                shuffle=True)

    # Reads the actual images from
    image, label = read_images_from_disk(input_queue)

    reshaped_image = tf.cast(image, tf.float32)
    # reshaped_label = tf.cast(label, tf.int32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_batch(float_image, label,
                                 min_queue_examples, batch_size,
                                 shuffle=False)


def _generate_image_batch(image, label, min_queue_examples,
                          batch_size, shuffle):
    """Construct a queued batch of images.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.contrib.deprecated.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])
