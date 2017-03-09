# coding:utf-8
import tensorflow as tf
import numpy as np
import threadpool, time,os
from multiprocessing import cpu_count


def section_com():
    file_paths = []
    for root, dirs, files in os.walk('/home/wangxiaopeng/operate/mid_files'):
        for file in files:
            # notice: read this file shoud strip '\n'
            file_paths.append(os.path.join(root, file))
    num = 1
    fw = open('/home/wangxiaopeng/NUS_dataset/mid_files/' + str(num) + '.txt', 'w')
    for i in file_paths:

        with open(i) as fr:

            fw.writelines(fr.readlines())
            fw.flush()
            if num % 5 == 0:
                fw.close()
                fw = open('/home/wangxiaopeng/NUS_dataset/mid_files/' + str(num) + '.txt', 'w')
        num += 1


if __name__ == '__main__':
    # file_paths = []
    # for root, dirs, files in os.walk('/home/wangxiaopeng/NUS_dataset/true_files'):
    #     for file in files:
    #         # notice: read this file shoud strip '\n'
    #         file_paths.append(os.path.join(root, file))
    #
    # fw = open('/home/wangxiaopeng/NUS_dataset/true_files/165208_true_file.txt', 'w')
    # for i in file_paths:
    #
    #     with open(i) as fr:
    #
    #         fw.writelines(fr.readlines())
    #         fw.flush()
    #     os.remove(i)
    # fw.close()

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/220341_pics_names.txt') as fw:
        a = set(fw.readlines())
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/gpu_220341_names.txt') as fw:
        b = set(fw.readlines())
    print len(a.intersection(b))