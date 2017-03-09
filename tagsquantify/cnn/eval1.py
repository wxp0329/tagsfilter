"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

import collections

with tf.Graph().as_default() as g:
    ones = tf.Variable(0 )
    zeros = tf.Variable(0 )
    Pair = collections.namedtuple('Pair', 'i')
    cons = tf.Variable(0,name='cons')
    img = tf.constant([1, 2, 3])


    size = tf.size(img)
    def whichroom(a,b):
        if a+b >3:
            return 0
        else:
            return 1
    def inner_cond(j,pj) :

        return (pj.i< 3)

    def add1(one_zero,sum1):
        one_zero = one_zero+sum1
        return np.array([one_zero])

    def inner_body(i,p):
        j = p.i
        # if img[i]==1:
        #      p.ones.append(img[i]+img[j])
        # else:
        #     p.zeros.append(img[i]+img[j])
        tf.cond(tf.equal(img[i],1) ,lambda :tf.assign_add(ones,tf.add(img[i],img[j])),lambda :tf.assign_add(zeros,tf.add(img[i],img[j])))
        j = j+1
        print j
        return (i, Pair(j))

    c = lambda i,j: i < size-1

    def outer_body(i,p):
        ijk_1 = (i, Pair(i+1))
        inn = tf.while_loop(inner_cond, inner_body, ijk_1)

        i = i + 1
        print i
        return inn

    ijk_0 = (tf.constant(0), Pair(tf.constant(0)))
    ijk_final = tf.while_loop(c, outer_body, ijk_0)

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())

        print s.run(ijk_final)
        print s.run(ones)
        print s.run(zeros)
