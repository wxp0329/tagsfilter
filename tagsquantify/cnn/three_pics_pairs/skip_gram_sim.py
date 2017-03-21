# coding:utf-8
import numpy as np
import threading
import threadpool, time, os
from multiprocessing import cpu_count


def str2arr(str_):
    arr = []
    for i in str_.strip().split(' '):
        arr.append(i)
    return np.array(arr, dtype=np.float)


def cosSimilar(inA, inB):
    num = np.sum(np.multiply(np.array(inA), np.array(inB)))
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def writefile(index):
    print '开始写第 ' + str(index) + ' 个文件了！'
    fw = open('/home/wangxiaopeng/test/' + str(index) + '_.txt', 'w')
    left_tag = tag_pair_list[index]
    # 判断该标签集在vec文件中是否存在
    ifexist = []
    for tag in left_tag[1]:
        ifexist.append(vec_pair.get(tag))
    # 如果vec文件中没有该图片所对应的任一值，则让这个pair的相似度=0
    if not any(ifexist):
        for i in xrange(index + 1, len(tag_pair_list)):
            fw.write(str(key_list_dict[left_tag[0]]) + ' ' + str(key_list_dict[tag_pair_list[i][0]]) + ' 0\n')
            fw.flush()
    # for i in xrange(index+1,len(tag_pair_list)):
    else:
        for i in xrange(index + 1, len(tag_pair_list)):
            right_tag = tag_pair_list[i]
            sum = 0.
            left_len = len(left_tag[1])
            right_len = len(right_tag[1])
            it = min(left_len, right_len)
            for j in xrange(it):
                left_vec = vec_pair.get(left_tag[1][j])
                right_vec = vec_pair.get(right_tag[1][j])

                if (left_vec == None) or (right_vec == None):
                    sum += 0.

                else:
                    sum += cosSimilar(str2arr(left_vec), str2arr(right_vec))  # //////right判断
            fw.write(str(key_list_dict[left_tag[0]]) + ' ' + str(key_list_dict[right_tag[0]]) + ' ' + str(
                round(sum / (left_len* right_len), 2)) + '\n')
            fw.flush()

    print '第 ' + str(index) + ' 个文件写完了。。。。。\n'
    fw.close()


if __name__ == '__main__':
    rootPath = '/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after'
    vec_pair = dict()
    with open(os.path.join(rootPath, '220341_tags_vec.txt')) as fr:
        for line in fr.readlines():
            ps = line.strip().split(' ', 1)
            vec_pair[ps[0]] = ps[1]
    print len(vec_pair.keys())

    tag_pair = dict()
    with open(os.path.join(rootPath, '220341_filter_tags.txt')) as fr:
        for line in fr.readlines():
            tags = []
            ps = line.strip().split(' ', 1)
            if len(ps) == 1:
                tags.append('#')
            else:
                for tag in ps[1].strip().split(' '):
                    tags.append(tag)

            tag_pair[ps[0]] = tags


    tag_pair_list = list(tag_pair.iteritems())

    key_list_dict = dict()
    with open(os.path.join(rootPath, '220341_key_list.dat')) as fr:
        num = 0
        for i in fr.readlines():
            key_list_dict[i.strip()] = num
            num += 1
    print len(tag_pair),len(key_list_dict)
    # print 'cpu_count :', cpu_count()
    # i_list = []
    # for i in xrange(len(tag_pair_list) - 1):  # 3000000 行大约50M数据
    #     i_list.append([i])
    #
    # n_list = [None for i in range(len(i_list))]
    # pool = threadpool.ThreadPool(cpu_count())
    # requests = threadpool.makeRequests(writefile, zip(i_list, n_list))
    # [pool.putRequest(req) for req in requests]
    # pool.wait()
    # print 'all of processes excute over !!!!!'
    writefile(2)
