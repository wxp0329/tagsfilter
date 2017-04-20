# coding:utf-8
import threadpool, time, os
from multiprocessing import cpu_count
import numpy as np


def generateTrueItemCount():
    a = dict()
    b = []
    with open('/home/wangxiaopeng/all_true_files.txt') as fr:
        for i in fr.readlines():
            i = i.strip().split(' ')
            b.append([i[0], i[1]])
            for item in i:

                if item in a.keys():
                    a[item] = a[item] + 1
                else:
                    a[item] = 1
                print a[item]
    fw = open('/home/wangxiaopeng/true_item_count.txt', 'w')
    sorted_fw = open('/home/wangxiaopeng/sorted_all_true_files.txt', 'w')
    for i in sorted(b, key=lambda x: x[0]):
        sorted_fw.write(i[0] + ' ' + i[1] + '\n')
        sorted_fw.flush()
    sorted_fw.close()
    for i in a.iteritems():
        fw.write(i[0] + ' ' + str(i[1]) + '\n')
        fw.flush()
        print i
    fw.close()



def write_file(left, right):

    fw = open('/home/wangxiaopeng/NUS_dataset/0.2_dir/' + str(left) + '_.txt', 'w')
    # key_list_np = np.array(key_list)
    for i in all_true_data[left:right]:
        line = i.strip().split(' ')
        count = int(line[1])
        items = set()
        i_key = part_tags[int(line[0])]
        i_set = all_line_dict[i_key]
        jaccard = 0.2
        j_list= []
        for j in key_list:
            j_set = all_line_dict[j]

            j_list.append((j,len(j_set.intersection(i_set))*1.0/len(j_set.union(i_set))))
        j_list = np.array(j_list)
        while True:
            np.random.shuffle(j_list)
            for pair in j_list:
                value = float(pair[1])
                if (value >= jaccard) and (value <= 0.2) :
                    items.add(pair[0] )
                    if len(items) == count:
                        for h in items:
                            fw.write(line[0] + ' ' + str(key_index[h]) + '\n')
                            fw.flush()
                        break
            if len(items) != count:
                jaccard -=0.01
                # print 'jaccard :',jaccard
            if len(items) == count:
                break

    print str(left) + '_.txt write over !!!\n'
    fw.close()

def run_write_files():
    global all_line_dict, all_true_data, part_tags, key_list, key_index
    part_tags = []
    with open('/home/wangxiaopeng/NUS_dataset/220341_key_list.dat') as fr:
        for i in fr.readlines():
            part_tags.append(i.strip())

    key_index = dict()
    c = 0
    for i in part_tags:
        key_index[i] = c
        c += 1

    all_line_dict = {}

    with open('/home/wangxiaopeng/NUS_dataset/replaced_220341_tags.dat') as fr:
        for line in fr.readlines():
            a = set()
            pic_labels = line.strip().split('__^__')
            if len(pic_labels) == 1:
                all_line_dict[pic_labels[0]] ='#'
                continue
            for label in pic_labels[1].strip().split(' '):
                if len(label) > 0:
                    a.add(label)
            all_line_dict[pic_labels[0]] = a

    key_list = all_line_dict.keys()

    # 记了数的true_files
    with open('/home/wangxiaopeng/NUS_dataset/true_item_count.txt') as fr:
        all_true_data = fr.readlines()

    print 'part len:', len(all_true_data)
    print 'cpu_count :', cpu_count()
    len_ = len(all_true_data)

    all_true_data = all_true_data[len_/2:] #my computer: [len_/2:]  gpu: [:len_/2]
    len_ = len_ -len_/2
    # all_true_data = all_true_data[:len_/2] #my computer: [len_/2:]  gpu: [:len_/2]
    # len_ = len_/2
    i_list = []
    for i in xrange(0, len_, 1000):

        if i + 1000 >= len_:
            i_list.append([i, len_ + 1])
            break
        i_list.append([i, i + 1000])
    n_list = [None for i in range(len(i_list))]
    pool = threadpool.ThreadPool(6)
    requests = threadpool.makeRequests(write_file, zip(i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print 'all of indexes excute over !!!!!'


def sort_false_file():#sored the false files the same order as sorted_all_true_files.txt
    b = []
    with open('/home/wangxiaopeng/all_false.txt') as fr:
        for i in fr.readlines():
            i = i.strip().split(' ')
            b.append([i[0], i[1]])

    sorted_fw = open('/home/wangxiaopeng/sorted_<0.2_files.txt', 'w')
    for i in sorted(b, key=lambda x: x[0]):
        sorted_fw.write(i[0] + ' ' + i[1] + '\n')
        sorted_fw.flush()
    sorted_fw.close()

def com():

    with open('/home/wangxiaopeng/NUS_dataset/test1.txt') as fr:
        test = fr.readlines()
    sorted_fw = open('/home/wangxiaopeng/0.2_three_pair.txt', 'w')
    c = 0
    with open('/home/wangxiaopeng/NUS_dataset/sorted_all_true_files.txt') as fr:
        for i in fr.readlines():
            sorted_fw.write(i.strip()+' '+test[c])
            sorted_fw.flush()
            c+=1
    sorted_fw.close()
if __name__ == '__main__':
    # run_write_files()
    # sort_false_file()
    com()