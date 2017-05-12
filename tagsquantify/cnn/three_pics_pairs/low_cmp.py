# coding:utf-8
import datetime
import numpy as np
import os, shutil
from multiprocessing import cpu_count

import threadpool


def genAll81Labels():
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels_name_list.txt') as fr:
        names = fr.readlines()
    a = []
    for i in names:
        a.append(np.reshape(
            np.loadtxt(os.path.join('/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels', i.strip()),
                       dtype=np.int64), [-1, 1]))

    new = np.concatenate(a, axis=1)
    print new.shape
    np.savetxt('/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels81', new, fmt='%i')


# 存的是2003个图片分别在81类中的索引，也就是每个Labels文件中值为1的位置（相对于269648）
def gen2003Index():
    ins = []
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            ins.append(int(i.strip().split(' ')[1]))
    allFlags = np.array([0] * 269648)
    allFlags[ins] = 1
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels'):
        for file in files:
            file_paths.append(os.path.join(root, file))
    for i in file_paths:
        a = np.loadtxt(i, dtype=np.int64)
        indis = np.argwhere(np.multiply(allFlags, a)).flatten()
        np.savetxt('/media/wangxiaopeng/maxdisk/NUS_dataset/2003_in_Allabels_index/len_'
                   + str(len(indis)) + '_' + str(i).rsplit('/', 1)[1], indis, fmt='%i')


# 把每个种类中的每个图片的特定低层特征向量与220341个特定的低层特征进行欧式距离计算得到top_count
# 把每个种类中的每个图片与其检索到的top_count个图片写入文件夹中
# 把该种类的每个元素所对应的检索到的top_count索引向量写入到文件
def generate_sim_pics(root_dir,
                      retrived_pics='/media/wangxiaopeng/maxdisk/NUS_dataset/2003_in_Allabels_index/len_1001_Labels_sky.txt',
                      feature='/media/wangxiaopeng/maxdisk/NUS_dataset/218838_Low_Level_Features/218838_BoW_int_500.dat'):
    accdir = os.path.join(root_dir, 'acc')
    mapdir= os.path.join(root_dir, 'map')
    recalldir = os.path.join(root_dir, 'recall')

    accexistFiles = []
    for root, dir, file in os.walk(accdir):
        accexistFiles = file

    mapexistFiles = []
    for root, dir, file in os.walk(mapdir):
        mapexistFiles = file

    recallexistFiles = []
    for root, dir, file in os.walk(recalldir):
        recallexistFiles = file

    if (retrived_pics.rsplit('_', 1)[1].split('.')[
        0] + '_map.txt' in accexistFiles) and (retrived_pics.rsplit('_', 1)[1].split('.')[
        0] + '_map.txt' in mapexistFiles) and (retrived_pics.rsplit('_', 1)[1].split('.')[
        0] + '_map.txt' in recallexistFiles):
        return 0




    index_names = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:

        for i in fr.readlines():
            pair = i.strip().split(' ')
            index_names[pair[1]] = pair[0]
    grass = []
    with open(retrived_pics) as fr:
        for i in fr.readlines():
            grass.append(i.strip())
    train_index_names = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_name_indexes.txt') as fr:

        for i in fr.readlines():
            pair = i.strip().split(' ')
            train_index_names[pair[1]] = pair[0]
    tfidf = np.loadtxt(feature)
    all_tfidf = np.loadtxt(
        '/media/wangxiaopeng/maxdisk/NUS_dataset/Low_Level_Features/' + feature.rsplit('/', 1)[1].split('_', 1)[1])
    print 'feature loading over !'
    # 每个pair存放的是[（索引，图片名），（索引，图片名）]
    pics_root_dir = '/home/wangxiaopeng/all_find_pics/find_pic_' + feature.rsplit('/', 1)[1].split('.')[0]
    # if os.path.exists(pics_root_dir):
    #     shutil.rmtree(pics_root_dir)
    # os.mkdir(pics_root_dir)

    # 开始查询部分
    # 存储每个图片对应的top_count中的索引
    fw = open(os.path.join(root_dir, 'time.txt'), 'a')
    pic_top_count = []
    start = datetime.datetime.now()
    for i in grass:
        pic_name_500 = index_names[i]
        # os.mkdir(os.path.join(pics_root_dir, pic_name_500))
        # with open('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220841' + pic_name_500 + '.jpg') as fr:
        #     with open(os.path.join(pics_root_dir, pic_name_500) + '/0_original_' + pic_name_500 + '.jpg', 'w') as fw:
        #         fw.writelines(fr.readlines())
        id = 1

        top_count_index = np.argsort(
            np.sqrt(np.sum(np.square(np.subtract(all_tfidf[int(i)], tfidf)), axis=1))).tolist()[:100]
        pic_top_count.append(top_count_index)
        # for ii in top_count_index:
        #     picName_220341 = train_index_names[ii]
        # with open('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220841' + picName_220341 + '.jpg') as fr:
        #     with open(os.path.join(pics_root_dir, pic_name_500) + '/' + str(id) + '_' + picName_220341 + '.jpg',
        #               'w') as fw:
        #         fw.writelines(fr.readlines())
        # id += 1
        # print pic_name_500, '的第', id, '个图片写完毕 !'
    end = datetime.datetime.now()
    consumeTime = (end - start).microseconds / 1000
    fw.write(retrived_pics.rsplit('_', 1)[1].split('.')[0] + ': ' + str(consumeTime) + '\n')

    # 把存有每个图片top_count的列表存入文件
    np.savetxt('/home/wangxiaopeng/top_count/cnn_top_counts' + feature.rsplit('/', 1)[1].split('_', 1)[1],
               np.array(pic_top_count), fmt='%i')


# 利用每个种类在500个样本测试集中的cnn映射向量，分别和220341个训练数据的生成向量比较欧式距离；
# 把每个种类对应的图片与检索到的相似的图片写入到一个文件夹
# 把该种类的每个元素所对应的检索到的top_count索引向量写入到文件
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
    mat_2003 = np.load('/home/wangxiaopeng/NUS_dataset/enforce_mats_500/2003_mat_48a.npy')
    mat_218838 = np.load('/home/wangxiaopeng/NUS_dataset/enforce_mats_500/combine_pic_mat_48a.npy')
    pics_root_dir = '/home/wangxiaopeng/all_find_pics/find_pic_cnn_' + retrived_pics.rsplit('_', 1)[1].split('.')[0]
    # if os.path.exists(pics_root_dir):
    #     shutil.rmtree(pics_root_dir)
    # os.mkdir(pics_root_dir)

    fw = open('/home/wangxiaopeng/lib_data_vgg/cnn_acc_map_recall_48a/cnn_topic_time.txt', 'a')
    # 开始查询部分
    # 存储每个图片对应的top_count中的索引
    pic_top_count = []
    start = datetime.datetime.now()
    count = 0
    for pic_index in category:
        pic_name = all_2003_name_index[pic_index.strip()]

        in_mat_index = key_2003_index[pic_name]
        to_all_dist = np.sqrt(np.sum(np.square(np.subtract(mat_2003[int(in_mat_index)], mat_218838)), axis=1))
        # to_all_dist = np.sum(np.fabs(np.subtract(mat_2003[int(in_mat_index)], mat_218838)), axis=1)
        order = 1
        #
        top_count_index = np.argsort(to_all_dist).tolist()[:100]
        pic_top_count.append(top_count_index)  # 保存
        # if count < 100:
        #     # 以该图片名建立目录并把该图片写入该目录
        #     os.mkdir(os.path.join(pics_root_dir, pic_name))
        #     with open(os.path.join('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220841', pic_name + '.jpg')) as fr:
        #         with open(os.path.join(pics_root_dir + '/' + pic_name, '0_original_' + pic_name + '.jpg'), 'w') as fw:
        #             fw.writelines(fr.readlines())
        #
        #     for i in top_count_index:
        #         with open(
        #                 os.path.join('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220841',
        #                              index_218838_name[i] + '.jpg')) as fr:
        #             with open(
        #                     os.path.join(pics_root_dir + '/' + pic_name,
        #                                  str(order) + '_' + index_218838_name[i] + '.jpg'),
        #                     'w') as fw:
        #                 fw.writelines(fr.readlines())
        #         order += 1
        #
        #     count += 1
    end = datetime.datetime.now()
    consumeTime = (end - start).microseconds / 1000
    fw.write(retrived_pics.rsplit('_', 1)[1].split('.')[0] + ': ' + str(consumeTime))
    # 把存有每个图片top_count的列表存入文件
    np.savetxt('/home/wangxiaopeng/top_count/cnn_top_counts', np.array(pic_top_count), fmt='%i')


############################################针对于单标签计算acc#######################################################################

def acc(top_count_category='/home/wangxiaopeng/top_count/cnn_top_counts',
        category_name='/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels/Labels_beach.txt'):
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/2003_in_Allabels_index/len_106_Labels_beach.txt') as fr:
        category1 = fr.readlines()
    category = np.loadtxt(category_name, dtype=np.int64)

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_key_list.txt') as fr:
        keys_220341 = np.array(fr.readlines())

    name_all_index = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_name_indexes.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_all_index[pair[0]] = pair[1]
    avg_acc = 0.
    top_count = np.loadtxt(top_count_category, dtype=np.int64)
    fw = open('/home/wangxiaopeng/acc.txt', 'w')
    max_min = []
    num = 0
    for i in top_count:  # i是每个图片对应的top_count index
        sum_element = len(i)
        every_avg_acc = 0.
        for j in keys_220341[i]:  # j是通过i得到的图片名列表
            index_in_all = name_all_index[j.strip()].strip()
            every_avg_acc += category[int(index_in_all)]
        every_avg_acc = every_avg_acc * 1.0 / sum_element
        max_min.append(every_avg_acc)
        print 'pic name:', all_2003_name_index[(category1[num]).strip()], every_avg_acc
        fw.write(str(all_2003_name_index[(category1[num]).strip()]) + ' ' + str(every_avg_acc) + '\n')
        avg_acc += every_avg_acc
        num += 1
    avg_acc /= len(top_count)  # 每个图片的准确率之和除以图片个数
    print 'max avg:', max(max_min)
    fw.write('max avg: ' + str(max(max_min)) + '\n')
    print 'min avg:', min(max_min)
    fw.write('min avg: ' + str(min(max_min)) + '\n')
    print 'acc :', avg_acc
    fw.write('acc :' + str(avg_acc) + '\n')
    fw.close()


############################################针对于单标签，两个图片比较仅考虑一个同样的标签#######################################################################
# sky：单标签准确率50%，多标签准确率

############################################针对于多标签标签，两个图片标签集有任意一个命中#######################################################################

def acc_multiLabels(retrived_pics, root_dir, top_count_category='/home/wangxiaopeng/top_count/cnn_top_counts',
                    all_category_name='/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/AllLabels81.txt'):



    with open(retrived_pics) as fr:
        retrives = fr.readlines()

    name_index_2003_in_all = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_index_2003_in_all[pair[1]] = pair[0]

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_key_list.txt') as fr:
        test_datas_name = fr.readlines()

    name_index_all_2003 = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_index_all_2003[pair[0]] = pair[1]

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_key_list.txt') as fr:
        keys_218838 = np.array(fr.readlines())

    name_all_index = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_name_indexes.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_all_index[pair[0]] = pair[1]

    top_count = np.loadtxt(top_count_category, dtype=np.int64)
    all_labels = np.loadtxt(all_category_name, dtype=np.int64)
    count = 0
    avg_all = 0.

    dir = os.path.join(root_dir, 'acc')
    if not os.path.exists(dir):
        os.makedirs(dir)
    fw = open(os.path.join(dir, retrived_pics.rsplit('_', 1)[1].split('.')[
        0] + '_acc.txt'), 'w')
    max_min = []
    for i in retrives:  # i是该图片在AllLabels81.txt中对应的索引
        every_avg = 0.
        pic_name = name_index_2003_in_all[i.strip()].strip()  # 获取查询图片名
        Rpic_81index = all_labels[int(i.strip())]  # 获取AllLabels81.txt中被查询图片的索引对应的行

        pic_top_count = top_count[count]  # 获取一个图片对应的topk个图片索引（相对于218838_key_list.txt的索引）
        for ind in pic_top_count:  # 遍历topk中的每个图片索引
            Tpic_in_all_index = name_all_index[keys_218838[ind].strip()]  # 获取topk中每个索引对应图片所在AllLabels.txt中的索引
            Tpic_81index = all_labels[int(Tpic_in_all_index)]  # 获取AllLabels.txt中的索引对应的行

            one_len = len(np.argwhere(np.multiply(Rpic_81index, Tpic_81index)).flatten())  # 计算查询图片与查询到的图片的标签集是否有交集
            if one_len > 0:
                every_avg += 1  # 记录topk中有几个与被查询图片的标签集有交集
        every_acc = every_avg * 1.0 / len(pic_top_count)  # 计算每个图片所查询到的准确率
        max_min.append(every_acc)
        fw.write(pic_name + ' ' + str(every_acc) + '\n')
        avg_all += every_acc  # 记录每个图片的准确率
        count += 1
    avg_all /= len(retrives)  # 计算所有图片的准确率的平均值

    fw.write('every_acc_min: ' + str(min(max_min)) + '\n')
    fw.write('every_acc_max: ' + str(max(max_min)) + '\n')
    fw.write('topic_avg_all: ' + str(avg_all) + '\n')
    fw.close()
    print retrived_pics.rsplit('_', 1)[1].split('.')[0], 'acc compute over!'
    return avg_all


def map_multiLabels(retrived_pics, root_dir, top_count_category='/home/wangxiaopeng/top_count/cnn_top_counts',
                    all_category_name='/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/AllLabels81.txt'):


    with open(retrived_pics) as fr:
        retrives = fr.readlines()

    name_index_2003_in_all = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_index_2003_in_all[pair[1]] = pair[0]

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_key_list.txt') as fr:
        test_datas_name = fr.readlines()

    name_index_all_2003 = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_index_all_2003[pair[0]] = pair[1]

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_key_list.txt') as fr:
        keys_218838 = np.array(fr.readlines())

    name_all_index = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_name_indexes.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_all_index[pair[0]] = pair[1]

    top_count = np.loadtxt(top_count_category, dtype=np.int64)
    all_labels = np.loadtxt(all_category_name, dtype=np.int64)
    count = 0
    map_all = []

    dir = os.path.join(root_dir, 'map')
    if not os.path.exists(dir):
        os.makedirs(dir)
    fw = open(os.path.join(dir, retrived_pics.rsplit('_', 1)[1].split('.')[
        0] + '_map.txt'), 'w')

    for i in retrives:  # i是该图片在AllLabels81.txt中对应的索引

        pic_name = name_index_2003_in_all[i.strip()].strip()  # 获取查询图片名
        Rpic_81index = all_labels[int(i.strip())]  # 获取AllLabels81.txt中被查询图片的索引对应的行

        pic_top_count = top_count[count]  # 获取一个图片对应的topk个图片索引（相对于218838_key_list.txt的索引）
        rank = []
        for ind in pic_top_count:  # 遍历topk中的每个图片索引
            Tpic_in_all_index = name_all_index[keys_218838[ind].strip()]  # 获取topk中每个索引对应图片所在AllLabels.txt中的索引
            Tpic_81index = all_labels[int(Tpic_in_all_index)]  # 获取AllLabels.txt中的索引对应的行

            ifSect = len(np.argwhere(np.multiply(Rpic_81index, Tpic_81index)).flatten())  # 计算查询图片与查询到的图片的标签集交集索引
            if ifSect > 0:
                rank.append(1)
            else:
                rank.append(0)

        # 计算每个检索图片的map值
        every_map = 0.
        order = 1
        for r in np.argwhere(np.array(rank)):
            every_map += order * 1.0 / (r + 1)
            order += 1
        if len(rank) == 0:
            every_map = 0.
        else:
            every_map = every_map * 1.0 / len(rank)
        map_all.append(every_map)  # 存放所有检索图片的map值
        fw.write(pic_name + ' ' + str(every_map) + '\n')
        count += 1

    topic_map = np.sum(np.array(map_all)) * 1.0 / len(map_all)  # 计算该主题的所有图片的平均map值
    fw.write('max_map: ' + str(max(map_all)) + '\n')
    fw.write('min_map: ' + str(min(map_all)) + '\n')
    fw.write('topic_avg_map: ' + str(topic_map) + '\n')
    fw.close()
    print retrived_pics.rsplit('_', 1)[1].split('.')[0], 'map compute over!'
    return topic_map


def recall_multiLabels(retrived_pics, root_dir, top_count_category='/home/wangxiaopeng/top_count/cnn_top_counts',
                       all_category_name='/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/AllLabels81.txt'):



    with open(retrived_pics) as fr:
        retrives = fr.readlines()

    name_index_2003_in_all = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_index_2003_in_all[pair[1]] = pair[0]

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_key_list.txt') as fr:
        test_datas_name = fr.readlines()

    name_index_all_2003 = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/2003_in_all_name_index.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_index_all_2003[pair[0]] = pair[1]

    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_key_list.txt') as fr:
        keys_218838 = np.array(fr.readlines())

    name_all_index = dict()
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/218838_in_all_name_indexes.txt') as fr:
        for i in fr.readlines():
            pair = i.strip().split(' ')
            name_all_index[pair[0]] = pair[1]

    top_count = np.loadtxt(top_count_category, dtype=np.int64)
    all_labels = np.loadtxt(all_category_name, dtype=np.int64)
    count = 0
    recall_all = []

    dir = os.path.join(root_dir, 'recall')
    if not os.path.exists(dir):
        os.makedirs(dir)
    fw = open(os.path.join(dir, retrived_pics.rsplit('_', 1)[1].split('.')[
        0] + '_recall.txt'),
              'w')

    for i in retrives:  # i是该图片在AllLabels81.txt中对应的索引

        pic_name = name_index_2003_in_all[i.strip()].strip()  # 获取查询图片名
        Rpic_81index = all_labels[int(i.strip())]  # 获取AllLabels81.txt中被查询图片的索引对应的行

        # 计算该查询图片在所有数据中的个数
        Rpic_count_compute = np.sum(np.multiply(np.array(Rpic_81index), all_labels), axis=1)  # 计算是否与all_labels有交集
        Rpic_relevant_count = len(np.argwhere(Rpic_count_compute))  # 得到与该检索图片有交集的图片个数

        pic_top_count = top_count[count]  # 获取一个图片对应的topk个图片索引（相对于218838_key_list.txt的索引）
        interSect = []
        for ind in pic_top_count:  # 遍历topk中的每个图片索引
            Tpic_in_all_index = name_all_index[keys_218838[ind].strip()]  # 获取topk中每个索引对应图片所在AllLabels.txt中的索引
            Tpic_81index = all_labels[int(Tpic_in_all_index)]  # 获取AllLabels.txt中的索引对应的行

            ifSect = len(np.argwhere(np.multiply(Rpic_81index, Tpic_81index)).flatten())  # 计算查询图片与查询到的图片的标签集交集索引
            if ifSect > 0:
                interSect.append(1)

        if Rpic_relevant_count == 0:
            every_recall = 0.
        else:
            every_recall = len(interSect) * 1.0 / Rpic_relevant_count  # 得到每个图片的召回率
        fw.write(pic_name + ' ' + str(every_recall) + '\n')
        recall_all.append(every_recall)
    avg_recall = np.sum(np.array(recall_all)) * 1.0 / len(recall_all)
    fw.write('topic_avg_recall: ' + str(avg_recall) + '\n')
    fw.close()
    print retrived_pics.rsplit('_', 1)[1].split('.')[0], 'recall compute over!'
    return avg_recall


############################################针对于多标签标签，两个图片标签集有任意一个命中#######################################################################

# 评测cnn提取特征
def cnn_acc_map_recall():
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/2003_in_Allabels_index/'):
        for file in files:
            file_paths.append(os.path.join(root, file))
    test_2003_avg = []
    test_2003_map = []
    test_2003_recall = []
    fw = open('/home/wangxiaopeng/lib_data_vgg/cnn_acc_map_recall_48a/all_2003_avg_acc.txt', 'w')
    fw_map = open('/home/wangxiaopeng/lib_data_vgg/cnn_acc_map_recall_48a/all_2003_avg_map.txt', 'w')
    fw_recall = open('/home/wangxiaopeng/lib_data_vgg/cnn_acc_map_recall_48a/all_2003_avg_recall.txt', 'w')
    root_dir = '/home/wangxiaopeng/lib_data_vgg/cnn_acc_map_recall_48a'
    for i in file_paths:
        cnn_gen_sim_pics(i)
        # 计算acc，recall，map
        category_pic_acc = acc_multiLabels(i, root_dir)
        category_pic_recall = recall_multiLabels(i, root_dir)
        category_pic_map = map_multiLabels(i, root_dir)

        test_2003_avg.append(category_pic_acc)
        test_2003_recall.append(category_pic_recall)
        test_2003_map.append(category_pic_map)

        fw.write(i.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0] + ': ' + str(category_pic_acc) + '\n')
        fw_recall.write(i.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0] + ': ' + str(category_pic_recall) + '\n')
        fw_map.write(i.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0] + ': ' + str(category_pic_map) + '\n')
    acc_2003_avg = np.sum(np.array(test_2003_avg, dtype=np.float)) / len(file_paths)
    acc_2003_recall = np.sum(np.array(test_2003_recall, dtype=np.float)) / len(file_paths)
    acc_2003_map = np.sum(np.array(test_2003_map, dtype=np.float)) / len(file_paths)
    fw.write('2003_avg_acc: ' + str(acc_2003_avg))
    fw_recall.write('2003_avg_recall: ' + str(acc_2003_recall))
    fw_map.write('2003_avg_map: ' + str(acc_2003_map))
    print 'all_2003_test avg_acc: ', acc_2003_avg
    print 'all_2003_test avg_recall: ', acc_2003_recall
    print 'all_2003_test avg_map: ', acc_2003_map
    fw.close()


# 评测低层特征
def low_feature_acc_map_recall(feature):
    file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/2003_in_Allabels_index/'):
        for file in files:
            file_paths.append(os.path.join(root, file))

    feature_dir = feature.rsplit('/', 1)[1].split('.')[0]
    root_dir = '/home/wangxiaopeng/lib_data/low_featrue_acc_map_recall/' + feature_dir
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    test_2003_avg = []
    test_2003_map = []
    test_2003_recall = []
    # fw = open(os.path.join(root_dir, 'all_2003_avg_acc.txt'), 'a')
    # fw_map = open(os.path.join(root_dir, 'all_2003_avg_map.txt'), 'a')
    # fw_recall = open(os.path.join(root_dir, 'all_2003_avg_recall.txt'), 'a')

    top_count_category = '/home/wangxiaopeng/top_count/cnn_top_counts' + feature.rsplit('/', 1)[1].split('_', 1)[1]
    for i in file_paths:
        ifnext = generate_sim_pics(root_dir,
                          retrived_pics=i
                          , feature=feature)
        if ifnext == 0:
            continue
        category_pic_acc = acc_multiLabels(i, root_dir, top_count_category=top_count_category)
        category_pic_recall = recall_multiLabels(i, root_dir, top_count_category=top_count_category)
        category_pic_map = map_multiLabels(i, root_dir, top_count_category=top_count_category)

        test_2003_avg.append(category_pic_acc)
        test_2003_recall.append(category_pic_recall)
        test_2003_map.append(category_pic_map)

        # fw.write(i.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0] + ': ' + str(category_pic_acc) + '\n')
        # fw_recall.write(
        #     i.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0] + ': ' + str(category_pic_recall) + '\n')
        # fw_map.write(i.rsplit('/', 1)[1].rsplit('_', 1)[1].split('.')[0] + ': ' + str(category_pic_map) + '\n')
    acc_2003_avg = np.sum(np.array(test_2003_avg, dtype=np.float)) / len(file_paths)
    acc_2003_recall = np.sum(np.array(test_2003_recall, dtype=np.float)) / len(file_paths)
    acc_2003_map = np.sum(np.array(test_2003_map, dtype=np.float)) / len(file_paths)
    # fw.write('2003_avg_acc: ' + str(acc_2003_avg))
    # fw_recall.write('2003_avg_recall: ' + str(acc_2003_recall))
    # fw_map.write('2003_avg_map: ' + str(acc_2003_map))
    print 'all_2003_test avg_acc: ', acc_2003_avg
    print 'all_2003_test avg_recall: ', acc_2003_recall
    print 'all_2003_test avg_map: ', acc_2003_map
    # fw.close()
    # fw_recall.close()
    # fw_map.close()


def threading_low_feature_acc_map_recall():
    low_file_paths = []
    for root, dirs, files in os.walk('/media/wangxiaopeng/maxdisk/NUS_dataset/218838_Low_Level_Features'):
        for file in files:
            low_file_paths.append([os.path.join(root, file)])
    print 'cpu_count :', cpu_count()

    len_ = len(low_file_paths)

    n_list = [None for ii in range(len_)]
    pool = threadpool.ThreadPool(3)
    requests = threadpool.makeRequests(low_feature_acc_map_recall, zip(low_file_paths, n_list))
    [pool.putRequest(req) for req in requests]
    pool.wait()
    print 'all of com excute over !!!!!'


if __name__ == '__main__':
    cnn_acc_map_recall()
    # threading_low_feature_acc_map_recall()
