# coding:utf-8

import numpy as np
import os
from collections import OrderedDict
import  tensorflow as tf
# encoding=utf-8


with tf.Session() as s:
    print s.run(tf.subtract(1,tf.constant([2,3,4,5])))