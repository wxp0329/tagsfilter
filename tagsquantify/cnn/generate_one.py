#coding:utf-8
import numpy as np
all_flags = []
with open('/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels/Labels_cars.txt') as fr:
    for names in fr.readlines():

        all_flags.append(names.strip())

all_flags1 = []
with open('/media/wangxiaopeng/maxdisk/NUS_dataset/Groundtruth/AllLabels/Labels_buildings.txt') as fr:
    for names in fr.readlines():
        all_flags1.append(names.strip())


a = np.array(all_flags,dtype=int)
b = np.array(all_flags1,dtype=int)

num = 0
for i in (a+b).tolist():
    if i == 2:
        print num
    num+=1
np.where()