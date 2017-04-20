# coding:utf-8
import Image
import numpy as np
import os
from collections import OrderedDict
import  tensorflow as tf
# encoding=utf-8
def getimg(str1):
    im = Image.open(str1)
    re_img = np.array(im.resize((60, 60)), dtype=np.float32)
    # Subtract off the mean and divide by the variance of the pixels.
    std = np.std(re_img)
    return np.divide(np.subtract(re_img, np.mean(re_img)), std)
pics = []
with open('/media/wangxiaopeng/maxdisk/NUS_dataset/tags/tags_after/220341_key_list.txt') as fr:
    for i in fr.readlines():
        pics.append(getimg(os.path.join('/media/wangxiaopeng/maxdisk/NUS_dataset/images_220841', i.strip() + '.jpg')))
np.save('/media/wangxiaopeng/maxdisk/NUS_dataset/218838_conv_pics',np.array(pics))

