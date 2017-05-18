# coding:utf-8

import numpy as np
from matplotlib  import pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Bitstream Vera Sans']

# X轴，Y轴数据

def map():
    # with open('/home/wangxiaopeng/found_pics_1/5group') as fr:
    #     # for i in  fr.readlines()[0] :
    #     al = fr.readlines()
    # y = []
    # for i in al[0].strip().split(' '):
    #     y.append(float(i) / 10.)
    # y1 = []
    # for i in al[1].strip().split(' '):
    #     y1.append(float(i) / 10.)
    # y2 = []
    # for i in al[2].strip().split(' '):
    #     y2.append(float(i) / 10.)
    # y3 = []
    # for i in al[3].strip().split(' '):
    #     y3.append(float(i) / 10.)
    # y4 = []
    # for i in al[4].strip().split(' '):
    #     y4.append(float(i) / 10.)
    py=[65.25,66.25,66.75,67.06,67.15]
    py1=[60.52,61.58,62.33,63.19,64.02]
    py2 =[53.40,55.39,56.38,57.38,59.33]
    py3 = [48.34,51.65,51.63,52.55,54.46]
    py4 = [45.53,46.44,46.51,47.17,47.49]
    py5=[40.48,41.58,42.56,43.34,45.28]


    y = [65.25,66.48,66.35,67.05,68.62]
    y1=[60.72,61.48,62.33,63.29,64.42]
    y2 = [53.40,55.39,56.38,57.68,59.43]
    y3 = [48.44,51.65,51.43,52.35,54.76]
    y4=[45.33,46.54,46.91,47.37,47.59]
    y5 = [40.48,41.48,42.36,43.44,45.88]
    x = [16,24,32,48,64]
    # y = [0.3,0.4,2,5,3,4.5,4]
    # y1 = [0.2,0.5,1,4,6,5.5,3]
    plt.figure(figsize=(5,5))  # 创建绘图对象
    # plt.subplot(121)
    # plt.plot(x, py, color="red", marker='x', ls='dashdot', linewidth=1, label='WSH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py1, color="green", marker='<', linewidth=1, label='DSRH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py2, color="blue", marker='h', linewidth=1, label='BRE-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py3, color="magenta", marker='|', linewidth=1, label='BRE')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py4, color="cyan", marker='d', linewidth=1, label='ITQ')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.plot(x, py5, color="black", marker='*', linewidth=1, label='LSH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    # plt.xticks(x, rotation=0)
    # plt.xlabel("#of bits")  # X轴标签
    # plt.ylabel("precision@10")  # Y轴标签
    # plt.subplot(122)
    plt.plot(x, y, color="red", marker='x', ls='dashdot', linewidth=1, label='WSH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y1, color="green", marker='<', linewidth=1, label='DSRH-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y2, color="blue", marker='h', linewidth=1, label='BRE-CNN')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y3, color="magenta", marker='|', linewidth=1, label='BRE')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y4, color="cyan", marker='d', linewidth=1, label='ITQ')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.plot(x, y5, color="black", marker='*', linewidth=1, label='LSH')  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

    plt.xticks(x, rotation=0)
    plt.xlabel("#of bits")  # X轴标签
    plt.ylabel("MAP@10")  # Y轴标签
    plt.legend(loc='upper right', bbox_to_anchor=(1.14, 1.14))
    # plt.grid(x)
    # plt.title("Precision")  # 图标题
    plt.show()  # 显示图, bbox_to_anchor=(1.29, 1.17)
    # plt.savefig("line.jpg") #保存图


def zhuDraw():
    n_groups = 2
    # fileter
    # Precision = (25.20, 23.59, 25.69)
    # recall = (78.29, 74.11, 79.71)
    # F1 = (38.13, 35.79, 38.86)
    # accuracy = (33.64, 28.85, 38.12)

    # without
    # Precision = (12.95,12.21,13.04)
    # recall = (78.90,74.72,80.33)
    # F1 = (22.25,20.99,22.43)
    # accuracy = (21.41,14.98,26.20)

    # # test
    # Precision = (29.64,35.35)
    # recall = (83.28,81.65)
    # F1 = (43.71,49.35)
    # accuracy = (69.42,78.66)

    #filter
    # s = (38.13,33.64)
    # s1 = (35.79,28.85)
    # my = (38.86,38.12)
    #without filter
    # s = (22.25, 21.41)
    # s1 = (20.99, 14.98)
    # my = (22.43, 26.20)
    #test
    s = (43.00, 73.42)
    s1 = (45.46,75.20)
    my = (47.48,79.54)
    my1 = (50.17,85.31)



    fig, ax = plt.subplots()

    # index = np.arange(n_groups)
    index = np.arange(n_groups)
    bar_width = 0.2

    opacity = 0.4

    # rects2 = plt.bar(index , s, bar_width, alpha=opacity, color='g', label=u'文献[46]的方法')
    # rects3 = plt.bar(index + bar_width , s1, bar_width, alpha=opacity, color='b', label=u'文献[47]的方法')
    # rects4 = plt.bar(index + bar_width + bar_width, my, bar_width, alpha=opacity, color='r',
    #                  label=u'本文的方法')

    rects2 = plt.bar(index, s, bar_width, alpha=opacity, color='g', label=u'VSM')
    rects3 = plt.bar(index + bar_width, my, bar_width, alpha=opacity, color='b', label=u'VSM+CNN')
    rects4 = plt.bar(index + bar_width + bar_width, s1, bar_width, alpha=opacity, color='g',
                     label=u'VSM+依存关系')
    rects5 = plt.bar(index + bar_width + bar_width+ bar_width, my1, bar_width, alpha=opacity, color='r',
                     label=u'本文的检索模型')
    # plt.xlabel(u'方法' )

    plt.ylabel(u'比率' )

    # plt.title('filtered candidate set',fontproperties=chinese_font)

    plt.xticks(index +0.3, (u'F1', u'准确率') )

    plt.ylim(0,100)

    # plt.legend(loc='upper right', bbox_to_anchor=(1.015,1.05))
    plt.legend(loc='upper right')


    plt.tight_layout()

    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
            # rect.set_edgecolor('white')

    # add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)
    add_labels(rects4)
    add_labels(rects5)
    plt.show()


def etract_220341_pairs():
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/filter_All_tags.dat') as fr:
        filter_pairs = np.array(fr.readlines(), dtype=np.string_)
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_in_all_only_indexes.txt') as fr:
        indexes = np.array(fr.readlines(), np.int)
    with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_filter_tags.txt', 'w') as fw:
        fw.writelines((filter_pairs[indexes]))

def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=float(inA*inB.T)
    denom=np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)

def cosSimilar1(inA,inB):
    num=np.sum(np.multiply(np.array(inA),np.array(inB)))
    denom=np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)
if __name__ == '__main__':
    map()
    # zhuDraw()