from keras import applications
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

model = applications.VGG16(include_top=False,weights='imagenet',input_shape=(60,60,3))
inps = np.load('/home/wangxiaopeng/NUS_dataset/218838_conv_pics.npy')
trans = model.predict(inps)
print 'trans shape:',trans.shape
np.save(open('/home/wangxiaopeng/NUS_dataset/218838_vgg16_pics.npy', 'w'),trans)
