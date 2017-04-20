# coding:utf-8
import tensorflow as tf
import numpy as np
import threadpool, time, os
from multiprocessing import cpu_count


# 从所有图片的标签中，抽取出220341个图片所对应的索引
def extractLabels():
    with open('/home/wangxiaopeng/NUS_dataset/tags_after/220841_in_all_only_indexes.txt') as fr:
        all_index = fr.readlines()
    all_index = np.array(all_index, dtype=np.int64)
    file_paths = []
    for root, dirs, files in os.walk('/home/wangxiaopeng/NUS_dataset/AllLabels'):
        for file in files:
            file_paths.append(os.path.join(root, file))
    for i in file_paths:
        np.savetxt('/home/wangxiaopeng/NUS_dataset/220841_AllLabels/220841_' + i.rsplit('/', 1)[1],
                   np.loadtxt(i, dtype=np.int64)[all_index], fmt='%i')
        print 'save over ', i

def count():
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/220841_AllLabels'):
        for file in files:
            file_paths.append(os.path.join(root, file))
    items = set()
    for i in file_paths:
        aa = np.argwhere(np.loadtxt(i, dtype=np.int64) == 1).flatten()
        # if len(aa) > 8472:
        print i.rsplit('/', 1)[1], len(aa)
        for item in aa:
            items.add(item)

    print len(items)


def pic_count_in_trueFile():
    a = set()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/sorted_all_true_files.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            a.add(pair[0])
            a.add(pair[1])
    print len(a)

def extractTest():
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/220841_AllLabels'):
        for file in files:
            file_paths.append(os.path.join(root, file))

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220841_key_list.txt') as fr:
        name_index =  np.array(fr.readlines())

    print len(name_index),'name_index'
    all_count = 171690
    all_picName = []
    for i in file_paths:
        aa = np.argwhere(np.loadtxt(i, dtype=np.int64) == 1).flatten()
        np.random.shuffle(aa)
        get_len = len(aa)*1.0/all_count*1300
        if get_len < 1:
            continue
        if get_len > 100:
            get_len = 100
            print i.strip().rsplit('/',1)[1], get_len
        all_picName.extend(name_index[ aa[:int(get_len)]])
    all_picName = set(all_picName)
    print len(all_picName)
    np.savetxt('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/test_picName.txt',np.array(list(all_picName), dtype=np.int64),fmt='%i')

def getSection():
    a = set()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/sorted_all_true_files.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            a.add(pair[0])
            a.add(pair[1])
    index_220341_name = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list.dat') as fr:
        count = 0
        for i in fr.readlines():
            index_220341_name[count] = i.strip()
            count += 1
    aa = set()
    for i in a:
        aa.add(index_220341_name[int(i)])
    b = set()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/test_picName.txt') as fr:
        for i in fr.readlines():
            b.add(i.strip())
    print 'a len',len(aa)
    print 'b len',len(b)
    print len(aa.intersection(b))

if __name__ == '__main__':
    # pic_count_in_trueFile()
    # getSection()
    extractTest()
    # count()