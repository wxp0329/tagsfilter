# encoding=utf-8
import threading

import datetime
import numpy as np
from nltk.corpus import wordnet as wn
import re, os


class JaccardTest:
    def __init__(self, root_path):
        self.root_path = root_path

    def synset_tag(self, tag):
        """把tag所对应的所有synset.lemma_names放到一个set中"""
        synset_tag = []

        print '正在求', tag, '标签的同义词集。。。。。。。。。。。。。。'
        try:
            for i in wn.synsets(str.strip(tag)):
                synset_tag.extend(i.lemma_names())
        except UnicodeDecodeError:
            return set()
        return set(synset_tag)

    def synset_tag_dict(self):
        """把每个标签及其对应的同义词集放入到dict中"""
        synset_tag_dict = dict()
        with open(self.root_path + 'filter_pure_tags.txt') as rf:
            all_nodup_tags = rf.readline().strip()
            for tag in all_nodup_tags.split(' '):
                tag = str.strip(tag)
                if len(tag) < 1:
                    continue
                synset_tag = self.synset_tag(tag)
                if len(synset_tag) > 0:
                    print '正在把', tag, '标签与其同义词集', synset_tag, '放到dict中。。。。。。。。。。。'
                    synset_tag_dict[tag] = synset_tag
        return synset_tag_dict

    def all_lines_tags(self):
        """# contains all not duplicated tags list,every line is all tags of
        # one imagine
        #注意：不包括图片编号
        # all_dup_tags.shape {1=[t1, t2, t3], 2=[t3, t5, t6]}"""
        all_dup_tags = {}
        print '读取所有的标签中......'
        with open(self.root_path + 'filter_All_tags_part1.dat') as all_tags:
            for line in all_tags.readlines():
                line_tags = []

                tags = re.split('\\s+', line, 1)
                if len(tags) < 1:
                    all_dup_tags[tags[0]] = ['###']
                    continue
                for tag in tags[1].strip().split(' '):
                    tag = str.strip(tag)
                    if len(tag) == 0:
                        continue
                    line_tags.append(tag)
                all_dup_tags[tags[0]] = line_tags
        print '读取所有标签,共', len(all_dup_tags), '个完毕！！！！！'
        return all_dup_tags

    def write_tag_replace(self, start, end, synset_tag_dict, raw_tags):
        """raw_tags:原始的标签集二维array，每一行代表一个图片的标签集{1=[t1, t2, t3], 2=[t3, t5, t6]}
        synset_tag_dict：每个标签对应的同义词集字典 标签=同义词集 ,方法synset_tag_dict()的返回值
        return：被替换后的二维标签set集，每一行代表一个图片的标签集"""
        raw_tags = list(raw_tags.iteritems())[start:end]
        for line_tags in raw_tags:  # line_tags:(1,[t1,t2])元组形式
            print '正在迭代原始的标签集的', line_tags, '!!!!!!!!!!!!!!...............'
            index = 0
            for tag in line_tags[1]:
                tag = str.strip(tag)
                for k, v in synset_tag_dict.iteritems():
                    if tag in v:
                        line_tags[1][index] = k
                        break
                index += 1
        # 消除每行的相同标签
        fw = open(self.root_path + '/jaccard_dataset/' + str(start) + '.dat', 'w')
        for line_tags in raw_tags:
            print line_tags, '被替换完毕。。。。！！！！！！！。。。。。。。。'
            fw.write(line_tags[0] + '__#__')
            for tag in set(line_tags[1]):
                fw.write(tag + ' ')  # 格式：1, 2, 3, 4
            fw.write('\n')
            fw.flush()
        fw.close()
        print '原始标签集被同义词替换写入到文件完毕！！！！！！！！！！！！'

    # 程序运行完毕后该上这一步读取所有替换好的标签集，把标签集放入方法save_pairs_Jaccard开始最后一步
    def read_tag_replace(self, filename):  # 改写!!!!!!!!!!!!!!!!
        """读取被替换好的每个图片的标签集
        :return格式： {pic1={a,bc,c},pic2={r,fe,d}}"""

        all_line_dict = {}

        with open(filename) as fr:
            for line in fr.readlines():
                a = set()
                pic_labels = line.strip().split('__#__')
                for label in pic_labels[1].strip().split(' '):
                    if len(label) > 0:
                        a.add(label)
                all_line_dict[pic_labels[0]] = a

        return all_line_dict
    def have_exists(self):
        e = []
        with open(self.root_path+'img_label_file.dat') as fr:
            for line in fr.readlines():
                e.append(line.strip().split(' ')[1])
        return e
    def save_pairs_Jaccard(self, data, delta=0.6):
        """计算每对图片标签集的Jaccard系数，如果大于等于delta，就把这对图片的index标记为1，否则为0
        data: read_tag_replace()的返回值，格式：{pic1={a,bc,c},pic2={r,fe,d}}"""
        data = list(data.iteritems())
        pairs_label = []
        have_exists = self.have_exists()
        fw = open(self.root_path + 'pairs_labels.dat', 'w')
        for i in range(len(data) - 1):
            if data[i][0] not in have_exists:
                continue
            for j in range(i + 1, len(data)):
                if data[j][0] not in have_exists:
                    continue
                print '正在比较图片', data[i], data[j]
                intersection = len(data[i][1].intersection(data[j][1]))
                union = len(data[i][1].union(data[j][1]))
                if union == 0:
                    continue
                jaccard = intersection * 1.0 / union
                if jaccard >= delta:
                    fw.write(data[i][0] + ' ' + data[j][0] + ' 1\n')
                    fw.flush()
                else:
                    fw.write(data[i][0] + ' ' + data[j][0] + ' 0\n')
                    fw.flush()
        fw.close()



if __name__ == '__main__':
    d1 = datetime.datetime.now()
    j = JaccardTest('/home/wxp/NSU_dataset/')
    # j.synset_tag_dict()
    # print j.tag_replace([['1','2','3','4','5'],['12','2','32','4','5']],{'b':['f'],'a':['12','32','43','3','5']})
    read_tag_replace = j.read_tag_replace('/home/wxp/NSU_dataset/replaced_All_tags.dat')
    j.save_pairs_Jaccard(read_tag_replace)
    # left = 0
    # i = 0
    # while 1:
    #
    #     right = 1000 * (i + 1)
    #     if right >= len(read_tag_replace):
    #         right = len(read_tag_replace) + 1
    #         th = threading.Thread(target=j.save_pairs_Jaccard, args=(read_tag_replace, left, right))
    #         th.start()
    #         break
    #
    #     th = threading.Thread(target=j.save_pairs_Jaccard, args=(read_tag_replace, left, right))
    #     th.start()
    #     left = right
    #     i += 1
    #
    d2 = datetime.datetime.now()
    print '一共用了', (d2 - d1).seconds, '秒！'
