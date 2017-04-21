# coding:utf-8
import datetime
import tensorflow as tf
import numpy as np
import threadpool, time, os
from multiprocessing import cpu_count


# 从所有图片的标签中，抽取出218838个图片所对应的索引
def extractLabels():
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_only_indexes.txt') as fr:
        all_index = fr.readlines()
    all_index = np.array(all_index, dtype=np.int64)
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels'):
        for file in files:
            file_paths.append(os.path.join(root, file))
    for i in file_paths:
        np.savetxt('/media/wangxiaopeng/maxdisk/NUS_dataset/218838_AllLabels/218838_' + i.rsplit('/', 1)[1],
                   np.loadtxt(i, dtype=np.int64)[all_index], fmt='%i')
        print 'save over ', i


def extractLowFeature():
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_only_indexes.txt') as fr:
        all_index = fr.readlines()
    all_index = np.array(all_index, dtype=np.int64)
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/Low_Level_Features'):
        for file in files:
            file_paths.append(os.path.join(root, file))
    for i in file_paths:
        if i == '/media/wangxiaopeng/maxdisk/NUS_dataset/Low_Level_Features/BoW_int_500.dat':
            np.savetxt(
                '/media/wangxiaopeng/maxdisk/NUS_dataset/218838_Low_Level_Features/218838_' + i.rsplit('/', 1)[1],
                np.loadtxt(i, dtype=np.int64)[all_index], fmt='%i')
        else:
            np.savetxt(
                '/media/wangxiaopeng/maxdisk/NUS_dataset/218838_Low_Level_Features/218838_' + i.rsplit('/', 1)[1],
                np.loadtxt(i, dtype=np.float)[all_index], fmt='%f')
        print 'save over ', i


def count():
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/218838_AllLabels'):
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
        name_index = np.array(fr.readlines())

    print len(name_index), 'name_index'
    all_count = 171690
    all_picName = set()
    key = set()
    for i in file_paths:
        clazz = i.strip().rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0]
        aa = np.argwhere(np.loadtxt(i, dtype=np.int64) == 1).flatten()
        np.random.shuffle(aa)
        get_len = len(aa) * 1.0 / all_count * 1294
        if get_len < 1:
            continue
        if get_len > 100:
            get_len = 100
            print i.strip().rsplit('/', 1)[1], get_len
        for e in name_index[aa[:int(get_len)]]:

            if e.strip() not in key:
                all_picName.add(e.strip() + ' ' + clazz)
            key.add(e.strip())

    print len(all_picName)
    np.savetxt('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/test_picName.txt', np.array(list(all_picName)), fmt='%s')


def gen218838NameIndex():
    a = set()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_test_picName.txt') as fr:
        for i in fr.readlines():
            a.add(i.strip().split(' ')[0])
    print len(a), '2003'
    fw = open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_name_indexes.txt', 'w')
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220841_in_all_name_indexes.txt') as fr:
        for i in fr.readlines():
            name = i.strip().split(' ')[0]
            if name not in a:
                fw.write(i)
    fw.close()


def getSection():
    b = set()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_test_picName.txt') as fr:
        for i in fr.readlines():
            b.add(i.strip().split(' ')[0])

    index_220341_name = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list.dat') as fr:
        count = 0
        for i in fr.readlines():
            index_220341_name[count] = i.strip()
            count += 1
    fw = open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_all_true_files.txt', 'w')
    a = set()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/sorted_all_true_files.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            if (index_220341_name[int(pair[0])] not in b) and (index_220341_name[int(pair[1])] not in b):
                fw.write(i)
    fw.close()


def genThreePair():
    b = set()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_key_list.txt') as fr:
        for i in fr.readlines():
            b.add(i.strip())
    fw = open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_0.2_three_pair.txt', 'w')

    index_220341_name = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list.dat') as fr:
        count = 0
        for i in fr.readlines():
            index_220341_name[count] = i.strip()
            count += 1
    a = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_0.2_three_pair.txt') as fr:
        for i in fr.readlines():
            cc = set()
            for ii in i.strip().split(' '):
                cc.add(index_220341_name[int(ii)])
            if len(cc.intersection(b)) == 0:
                fw.write(i)
    fw.close()


def gen2003NameIndex():
    b = []
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_test_picName.txt') as fr:
        for i in fr.readlines():
            b.append(i.strip().split(' ')[0])
    fw = open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt', 'w')
    a = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220841_in_all_name_indexes.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            a[pair[0]] = pair[1]
    for i in b:
        fw.write(i + ' ' + a[i] + '\n')
    fw.close()


def which_feature():
    low_file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/Low_Level_Features'):
        for file in files:
            low_file_paths.append(os.path.join(root, file))
    for i in low_file_paths:
        print i.rsplit('/', 1)[1]
        with open(i) as fr:
            print len(fr.readline().strip().split(' '))


def cnn_gen_sim_pics(retrived_pics):
    global all_2003_name_index
    with open(retrived_pics) as fr:
        category = fr.readlines()

    all_2003_name_index = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            all_2003_name_index[pair[1]] = pair[0]

    key_2003_index = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_key_list.txt') as fr:
        count = 0
        for i in fr.readlines():
            key_2003_index[i.strip()] = count
            count += 1

    index_218838_name = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_key_list.txt') as fr:
        count = 0
        for i in fr.readlines():
            index_218838_name[count] = i.strip()
            count += 1
    mat_2003 = np.load('/home/wangxiaopeng/NUS_dataset/enforce_mats_500/2003_hash_mat.npy')
    mat_218838 = np.load('/home/wangxiaopeng/NUS_dataset/enforce_mats_500/combine_hash_pic_mat.npy')
    pics_root_dir = '/home/wangxiaopeng/all_find_pics/find_pic_cnn_' + retrived_pics.rsplit('_', 1)[1].split('.')[0]
    # if os.path.exists(pics_root_dir):
    #     shutil.rmtree(pics_root_dir)
    # os.mkdir(pics_root_dir)

    fw = open('/home/wangxiaopeng/lib_data/cnn_acc_map_recall/cnn_topic_time_hanming_xor.txt', 'a')
    # 开始查询部分
    # 存储每个图片对应的top_count中的索引
    pic_top_count = []
    start = datetime.datetime.now()
    count = 0
    for pic_index in category:
        pic_name = all_2003_name_index[pic_index.strip()]

        in_mat_index = key_2003_index[pic_name]
        # to_all_dist = np.sqrt(np.sum(np.square(np.subtract(mat_2003[int(in_mat_index)], mat_218838)), axis=1))
        to_all_dist = np.sum((np.subtract(mat_2003[int(in_mat_index)], mat_218838)), axis=1)
        order = 1
        #
        top_count_index = np.argsort(to_all_dist).tolist()[:100]
        pic_top_count.append(top_count_index)  # 保存

    end = datetime.datetime.now()
    consumeTime = (end - start).microseconds / 1000
    fw.write(retrived_pics.rsplit('_', 1)[1].split('.')[0] + ': ' + str(consumeTime) + '\n')
    fw.flush()
    fw.close()


def get_low_acc_map_recall():
    content = ['sky', 'mountain', 'castle', 'valley', 'lake', 'sunset', 'ocean', 'clouds', 'rainbow', 'harbor']

    fw = open('all_acc.txt','w')

if __name__ == '__main__':
    get_low_acc_map_recall()

