# coding:utf-8


import threadpool, time, os
from multiprocessing import cpu_count

all_line_dict = {}

with open('/home/wangxiaopeng/replaced_220341_tags.dat') as fr:
    for line in fr.readlines():
        a = set()
        pic_labels = line.strip().split('__^__')
        for label in pic_labels[1].strip().split(' '):
            if len(label) > 0:
                a.add(label)
        all_line_dict[pic_labels[0]] = a

key_list = all_line_dict.keys()


def write_files(i):
    print 'start write:', i, '..........'

    fw = open('/home/wangxiaopeng/operate/mid_files/' + str(i) + '_mid_file.txt', 'w')
    i_key = key_list[i]
    i_set = all_line_dict[i_key]
    for j in key_list[i + 1:]:
        j_set = all_line_dict[j]
        if len(j_set.intersection(i_set)) == 0:
            fw.write(i_key + ' ' + j + '\n')
            fw.flush()
    fw.close()
    print 'save :', i, 'over  !!!!!!!!!!!!'


if __name__ == '__main__':#GPU 机器写到了第18138和剩余的图片比较所得的交集，下次该 i = 18139  !!!!
    print 'cpu_count :', cpu_count()

    i_list = [[i] for i in xrange(0, len(key_list))]

    n_list = [None for i in xrange(len(i_list))]

    pool = threadpool.ThreadPool(cpu_count())
    requests = threadpool.makeRequests(write_files, zip(i_list, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print 'all of intersection processes excute over !!!!!'
