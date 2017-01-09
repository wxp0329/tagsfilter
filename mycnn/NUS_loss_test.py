# coding=UTF-8
import numpy as np
import os
from PIL import Image
import CNN_Layer


class NUSDataTrain:
    IMG_SIZE = 240
    Y = None
    def __init__(self, rootDir='/home/wxp/NSU_dataset', pair_labels_name='256_pairs_labels.dat'):
        self.rootDir = rootDir
        self.pair_labels = self.read_labels(pair_labels_name)

    # 文件格式：lable1 label2 flag
    def read_labels(self, filename):
        all_pair_labels = []
        with open(os.path.join(self.rootDir, filename)) as fr:
            for line in fr.readlines():
                elements = line.strip().split(' ')
                all_pair_labels.append(elements)
        return all_pair_labels

    # 判断label_pair 属于那个类
    def which_room(self, labels, i, j):
        print('which_room')
        pair_labels = {labels[i], labels[j]}
        for i in self.pair_labels:
            if len(pair_labels.intersection(set(i))) == 2:
                return i[2]  # 1 or 0

    def loss(self, x, y, lamda=1):
        """
        """
        # Calculate the average cross entropy loss across the batch.
        print 'operate myself custom loss.................'
        NUSDataTrain.Y = y
        labels_size = len(y)
        one_labels = []
        zero_labels = []
        for i in xrange(labels_size - 1):
            for j in xrange(i + 1, labels_size):
                pair_sub = np.subtract(x[i], x[j])
                pair_square = np.square(pair_sub)
                pair_sum = np.sum(pair_square)
                if self.which_room(y, i, j):

                    one_labels.append(pair_sum)
                else:

                    zero_labels.append(np.maximum(0, lamda - pair_sum))

        loss_my = np.sum(one_labels) + np.sum(zero_labels)

        return loss_my

    def eval_numerical_gradient(self, f, w):
        """    一个f在x处的数值梯度法的简单实现
        - f是只有一个参数的函数
        - x是计算梯度的点
        """
        print 'compute NUS loss grad...............'
        grad = np.zeros(w.shape)
        h = 0.00001
        # 对w中所有的索引进行迭代
        it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            # 计算x+h处的函数值
            ix = it.multi_index
            old_value = w[ix]
            w[ix] = old_value + h  # 增加h
            fxh = f(w)  # 计算f(x + h)
            w[ix] = old_value - h  # 减去h
            fx_h = f(w)  # 计算f(x - h)
            w[ix] = old_value  # 存到前一个值中 (非常重要，只是计算梯度，最后需要把数据恢复原状)
            # 计算偏导数（中心差值公式）
            grad[ix] = (fxh - fx_h) / (2 * h)  # 坡度
            it.iternext()  # 到下个维度

            print 'compute NUS loss grad over !!!!!!!!!!!!!!!!!!!!...............'
            return grad

    def grad_loss(self, x):

        return self.loss(x, NUSDataTrain.Y)

    def read_labeled_image_list(self, image_list_file):
        """
        Read a .txt file containing pathes and labeles.
        Parameters
        ----------
         image_list_file : a .txt file with one /path/to/image per line
         label : optionally, if set label will be pasted after each line
        Returns
        -------
           List with all filenames in file image_list_file
        """
        f = open(image_list_file, 'r')
        filenames = []
        labels = []
        for line in f:
            filename, label = line.strip().split(' ')
            filenames.append(os.path.join(os.path.join(self.rootDir, '256_images'), filename))
            labels.append(label)
        return filenames, labels

    def read_images_from_disk(self, input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Parameters
        ----------
          filename_and_label_tensor: A scalar string tensor.
        Returns
        -------
          Two tensors: the decoded image, and the string label.
        """
        print 'loading data ..................'
        labels = input_queue[1]
        examples = []
        for i in input_queue[0]:
            img = Image.open(i)
            img1 = img.crop((0, 0, NUSDataTrain.IMG_SIZE, NUSDataTrain.IMG_SIZE))
            examples.append(np.array(img1).transpose(2, 1, 0))
        print 'loaded data size :',len(examples),''
        return np.array(examples,dtype='float32'), np.array(labels)

if __name__ == '__main__':
    ndt = NUSDataTrain()
    x, y = ndt.read_images_from_disk(ndt.read_labeled_image_list('/home/wxp/NSU_dataset/256_img_label_file.dat'))
    CNN_Layer.CNNLayer().NUStrain(x, y, batch_size=20)
