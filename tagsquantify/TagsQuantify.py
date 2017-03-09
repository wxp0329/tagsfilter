# encoding=utf-8
import codecs
import threading

import numpy as np
from sklearn import feature_extraction
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import random
from sklearn.cluster import KMeans
import sys, os


class TagsQuantify:
    def __init__(self, rootname):
        # '/home/wxp/NSU_dataset/'
        # load and return tfidf of SIFT wob of all imagines
        self.rootname = rootname
        print '加载数据中.......'
        self.tfidfs = np.loadtxt(self.rootname + 'BoW_int_tfidf.dat', delimiter=' ')
        self.all_dup_tags = self.all_dup_tags()
        self.all_nodup_tags = self.all_nodup_tags()
        self.img_index_tags = self.get_img_index_tags()
        print '数据加载完毕!!!!!!!'
        self.kmeans = KMeans(1).fit(self.tfidfs)  # 对总体tfidfs进行聚类

    def bowtfidf(self, filename):
        allLines = []
        for line in open(filename):
            allLines.append(str.strip(line))
        print len(allLines)
        vectors = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tf_idf = transformer.fit_transform(
            vectors.fit_transform(allLines))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        weights = tf_idf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        print type(weights)
        np.savetxt(self.rootname + 'BoW_int_tfidf.dat', weights, delimiter=' ')

    def all_dup_tags(self):
        """# contains all not duplicated tags list,every line is all tags of
        # one imagine
        #注意：不包括图片编号
        # all_dup_tags.shape [[t1, t2, t3], [t3, t5, t6]]"""
        all_dup_tags = []
        print '读取所有的标签中......'
        with open(self.rootname + 'All_Tags.txt') as all_tags:
            for line in all_tags.readlines():
                line_tags = []
                tags = re.split('\\s+', line, 1)[1]
                for tag in tags.strip().split(' '):
                    line_tags.append(str.strip(tag))
                all_dup_tags.append(line_tags)
        print '读取所有标签,共', len(all_dup_tags), '个完毕！！！！！'
        return all_dup_tags

    def all_nodup_tags(self):
        # contains all not duplicated tags set, finally return the set
        all_nodup_tags = set()
        print '不重复读取所有的标签中......'
        with open(self.rootname + 'pure_tags_part1.txt', 'r') as pure_tags:
            strs = str(pure_tags.readline()).strip().split(' ')
            for s in strs:
                s = s.strip()
                if len(s) == 0:
                    continue
                all_nodup_tags.add(s)
        print '不重复读取所有标签', len(all_nodup_tags), '完毕！！！！'
        return all_nodup_tags

    def get_img_index_tags(self):
        """return all_tag_img_indexes：每个标签=所对应的图片index集"""
        print '正在读取所有图片及其标签集............'
        # 文件格式：tag__#__1, 3, 5, 6
        all_tag_img_indexes = dict()
        with open(self.rootname + 'all_tag_indexes.dict') as fr:
            for line in fr.readlines():
                items = str(line).strip().split('__#__')
                elements = []
                for element in items[1].strip().split(', '):  # 1, 3, 5, 6
                    elements.append(element)
                all_tag_img_indexes[items[0]] = elements
        return all_tag_img_indexes

    def get_tfidf_by_indexes(self, indexes):  # !!!!!!!!!!!!!!!!!!!indexes==0的问题！！
        """获取图片角标所对应的tfidf集合
        :return : numpy tpye"""
        print '正在获取图片角标所对应的tfidf集合.............'
        tfidfs_indexes = []
        print 'indexes:', len(indexes)
        for i in indexes:
            tfidfs_indexes.append(self.tfidfs[int(i), :])
        print '图片角标所对应的tfidf集合获取完毕!!!!!!!!!!!'
        return np.array(tfidfs_indexes)

    def random_indexes(self, indexes_len):
        """随机生成|It|（即indexes_len）个角标"""
        print '随机生成', len(indexes_len), '个角标........'
        random_indexes = set()
        while 1:
            random_indexes.add(random.randint(0, self.tfidfs.shape[0] - 1))
            if len(random_indexes) == indexes_len:
                break
        return random_indexes

    def random_label(self):
        """随机产生一个标签
        :return 字符串"""
        return list(self.all_nodup_tags)[random.randint(0, len(self.all_nodup_tags) - 1)]

    def get_Phi(self, label):
        """获取label的kmeans 的 inertia_ /|It|（inertia_：Sum of distances of samples to their closest cluster center）"""
        print '计算', label, '的get_Phi...........'
        all_tag_img_indexes = self.img_index_tags  # 每个标签=所对应的图片index集
        tag_img_indexes = all_tag_img_indexes[str.strip(label)]
        label_tfidfs = self.get_tfidf_by_indexes(tag_img_indexes)  # 该标签所对应的tfidfs集合

        kmeans = KMeans(1).fit(label_tfidfs)
        center = kmeans.cluster_centers_  # array类型
        Phi = kmeans.inertia_ / len(tag_img_indexes)  # 该标签内聚距离和
        print '计算', label, '的get_Phi完毕!!!!!!!!!!!'
        return Phi, center

    def get_Psi(self, label_center):
        """label_center: Cent(It) or Cent(Irand) array类型
        获取该标签对应的tfidf集中心向量到总体tfidfs中心向量的cosine距离"""
        print '计算中心点的get_Psi...........'
        kmeans = self.kmeans  # 对总体tfidfs进行聚类
        print '计算中心点的get_Psi完毕!!!!!!!!!'
        return sklearn.metrics.pairwise.cosine_similarity(label_center, kmeans.cluster_centers_)

    def del_tag_compute(self, start, end):
        """start与end参数方便多线程的使用"""

        fw = open(self.rootname + 'deltags/deltags_' + str(start) + '.dat', 'w')
        for tag in list(self.all_nodup_tags)[start:end]:
            tag = str.strip(tag)
            if len(tag) < 1:
                continue

            print '第', str(start), '线程判断 ', tag, ' 是否合格.........'
            tag_Phi, tag_center = self.get_Phi(tag)
            random_tag_Phi, random_tag_center = self.get_Phi(self.random_label())
            if tag_Phi > random_tag_Phi:
                fw.write(tag + ' ')
                fw.flush()
                continue
            elif self.get_Psi(tag_center) < self.get_Psi(random_tag_center):
                fw.write(tag + ' ')
                fw.flush()
        fw.close()

    def tag_indexes_dict(self, start, end):
        """start与end形式容易使用多线程
        功能：把标签和包含它的所有图片形成tag=[index1,index2]"""
        print '第', start, '个线程正在读取所有图片及其标签集............'

        all_tags = list(self.all_nodup_tags)
        tagnum = 0
        fw = open('/home/wxp/NSU_dataset/tag_indexes/' + str(start) + 'tag_indexes.dat', 'w')
        for tag in all_tags[start:end]:  # 读取图片标签集合
            tagnum += 1
            num = 0  # index
            tag_img_indexes = []  # 存放包含该标签的图片index
            print '该上第', start, '线程的', tag, '标签了。。。。。'
            tag = str.strip(tag)
            if len(tag) == 0:
                continue
            for tags in self.all_dup_tags:  # 每行代表一个图片的所有标签

                if tag in tags:
                    tag_img_indexes.append(num)
                num += 1

                # if (len(tag) > 0) and (len(tag_img_indexes) > 0):
                # all_tag_img_indexes[tag] = tag_img_indexes#
            if len(tag_img_indexes) > 0:
                print '写入了文件。。。。。。。。。。。。。。。。。'
                fw.write(tag + '__#__' + str(tag_img_indexes).strip('[]') + '\n')  # 文件格式：tag__#__1, 3, 5, 6
                fw.flush()
                # os.fsync(fw)
        fw.close()

    # 接下来这一步解决了就开始输出所有deltags分文件了！！！！
    # 接着利用这个方法把deltags分文件合并为一个大文件！！！！
    # 接着山除掉All_tags.dat文件里的deltags字符，过滤标签搞定！！2337587522
    def combine_tag_indexes_dict(self):
        """把用多线程写的tag=indexes的分文件合成为一个大文件"""
        file_paths = []
        for root, dirs, files in os.walk('/home/wxp/NSU_dataset/tag_indexes/'):
            for file in files:
                file_paths.append(os.path.join(root, file))
        fw = open('/home/wxp/NSU_dataset/all_tag_indexes1.dict', 'w')
        for file in file_paths:
            with open(file) as fr:
                for line in fr.readlines():
                    fw.write(line + '\n')
                    fw.flush()
        fw.close()


def del_all_tags():
    # 读取deltags_all.dat文件
    del_list = []
    print '读取deltags_all.dat文件....................'
    with open('/home/wxp/NSU_dataset/deltags_all.dat') as fr:
        for i in fr.readline().strip().split(' '):
            del_list.append(i)

    # 读取All_tags.dat文件
    print '读取All_tags.dat文件....................'
    fw = open('/home/wxp/NSU_dataset/filter_All_tags.dat','w')
    with open('/home/wxp/NSU_dataset/All_tags.dat') as fr:#..............
        for line in fr.readlines():
            tags=[]
            pic_tags = re.split('\\s+', line.strip(), 1)
            if len(pic_tags) < 2:
                fw.write(pic_tags[0]+'\n')
                continue
            for tag in pic_tags[1].strip().split(' '):
                tags.append(tag)
            copy_tags = []
            copy_tags.extend(tags)
            for i in copy_tags:#不能边迭代边删除，做一个copy_tags即可
                if i in del_list:
                    print 'remove :',i,'........................'

                    tags.remove(i)
            fw.write(pic_tags[0]+" ")
            for i in tags:
                fw.write(i+' ')
            fw.write('\n')
            fw.flush()
    fw.close()

if __name__ == '__main__':
    # del_tag_compute()
    # tq = TagsQuantify('/home/wxp/NSU_dataset/')
    # a = list(tq.all_nodup_tags)
    #
    # left = 0
    # i = 0
    # while 1:
    #
    #     right = 100 * (i + 1)
    #     if right >= len(a):
    #         right = len(a) + 1
    #         th = threading.Thread(target=tq.del_tag_compute, args=(left, right))
    #         th.start()
    #         break
    #
    #     th = threading.Thread(target=tq.del_tag_compute, args=(left, right))
    #     th.start()
    #     left = right
    #     i += 1
    del_all_tags()