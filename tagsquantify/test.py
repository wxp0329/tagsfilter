# coding:utf-8
from imghdr import what

import tensorflow as tf
import numpy as np
import os

# encoding=utf-8
import datetime
def read_tag_replace(filename):  # 改写!!!!!!!!!!!!!!!!
    """读取被替换好的每个图片的标签集
    :return格式： {pic1={a,bc,c},pic2={r,fe,d}}"""

    all_line_dict = {}
    # fw1 = open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list_dat', 'w')
    with open(filename) as fr:
        for line in fr.readlines():
            a = set()
            pic_labels = line.strip().split('__^__')
            for label in pic_labels[1].strip().split(' '):
                if len(label.strip()) > 0:
                    a.add(label)
            # fw1.write(pic_labels[0]+'\n')
            all_line_dict[pic_labels[0]] = a
    print 'dict len： ',len (all_line_dict)
    return all_line_dict

def get_key_list():
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list_dat') as fr:
        return fr.readlines()

all_dict = read_tag_replace('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/replaced_220341_tags.dat')
fw = open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/intersect_replaced_220341_tags.dat','w')
key_list = get_key_list()
len_ = len(key_list)
start = datetime.datetime.now()
for i in range(len_ -1):
    for j in range(i+1,len_ ):
        i_key = key_list[i]
        j_key = key_list[j]
        i_set = all_dict[i_key.strip()]
        j_set = all_dict[j_key.strip()]
        di = (len(set(j_set).union(i_set))*1.0)
        if di==0:
            continue
        num = round(len((j_set).intersection(i_set))/di,2)
        if (num > 0):
            fw.write(str(i)+' '+str(j)+' '+str(num)+'\n')

            print '写入 ： '+str(i) + ' ' + str(j) + ' ' + str(num)
    fw.flush()
fw.close()
end = datetime.datetime.now()
print '用时： '+str((end-start).seconds)