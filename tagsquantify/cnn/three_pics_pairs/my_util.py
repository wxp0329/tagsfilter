# coding:utf-8
import numpy as np
import threading

import shutil
import threadpool, time, os
from multiprocessing import cpu_count

def check_pic(yuzhi_file):
    rootPath = '/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after'
    imagePath = '/media/wangxiaopeng/maxdisk/NUS_dataset/images_220341'

    write_pic_dir = '/home/wangxiaopeng/find_pic'
    if os.path.exists(write_pic_dir):
        shutil.rmtree(write_pic_dir)
    os.mkdir(write_pic_dir)
    with open(os.path.join(rootPath, '220341_key_list.dat')) as fr:
        key_list = fr.readlines()
    num = 0
    with open(yuzhi_file) as fr:
        for i in fr.readlines():
            line = i.strip().split(' ')
            fileName1 = key_list[int(line[0])] + '.jpg'
            fileName2 = key_list[int(line[1])] + '.jpg'
            with open(os.path.join(imagePath, fileName1)) as fr:
                with open(os.path.join(write_pic_dir, str(num) + '_' + fileName1), 'w') as fw:
                    fw.writelines(fr.readlines())

            with open(os.path.join(imagePath, fileName2)) as fr:
                with open(os.path.join(write_pic_dir, str(num) + '_' + fileName2), 'w') as fw:
                    fw.writelines(fr.readlines())
            num += 1
#按照大小阈值分堆
def write_files(i):
    print 'start write file:', str(i), '...........'
    true_fw = open('/home/wangxiaopeng/NUS_dataset/three_pair_dir/three_true_files/' + str(i) + '_.txt','w')
    false_fw = open('/home/wangxiaopeng/NUS_dataset/three_pair_dir/three_false_files/' + str(i) + '_.txt','w')
    with open('/home/wangxiaopeng/NUS_dataset/three_pair_dir/all_files/'+str(i)+'_.txt') as fr:
        for pair in fr.readlines():
            pair = pair.strip().split(' ')
            if float(pair[-1]) >= 0.4:
                true_fw.write(pair[0] + ' ' + pair[1] + '\n')
                true_fw.flush()
            elif float(pair[-1]) < 0.2:
                false_fw.write(pair[0] + ' ' + pair[1] + '\n')
                false_fw.flush()

    true_fw.close()
    false_fw.close()
    print 'write file:', str(i), 'over !!!!!!!!!!!'


def run_write_true_false_files():

    print 'cpu_count :', cpu_count()
    i_list = []
    for i in xrange(220330):

        i_list.append([i])
    n_list = [None for i in range(len(i_list))]
    pool = threadpool.ThreadPool(cpu_count())
    requests = threadpool.makeRequests(write_files, zip(i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print 'all of processes excute over !!!!!'
#合并为三元组
def combTriple(i):
    with open('/home/wangxiaopeng/three_true_files/' + str(i) + '_.txt') as fr:
        i_true_file = fr.readlines()
    with open('/home/wangxiaopeng/three_false_files/' + str(i) + '_.txt') as fr:
        i_false_file = fr.readlines()
    com_file = open('/home/wangxiaopeng/three_false_files/com_files/' + str(i) + '_com_.txt','w')
    len_ = min(len(i_true_file),len(i_false_file))

if __name__ == '__main__':
    # compute_sim()
    check_pic('')
