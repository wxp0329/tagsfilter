# encoding = utf-8
import cPickle,os
import numpy as np
from PIL import Image
import layers
import softmax_test
import optim
import matplotlib.pyplot as plt


def load_CIFAR101(cifar10_dir):
    file_paths = []
    for root, dirs, files in os.walk(cifar10_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    all_trian_datas = []
    all_trian_labels = []

    for i in file_paths:
        if not str(i).startswith(cifar10_dir+'/data_batch'):
            continue
        fo = open(i)
        dict = cPickle.load(fo)
        fo.close()
        # train image datas shape is : (10000L, 3072L)
        datas = dict['data']
        # image labels :(10000L,)
        labels = dict['labels']
        all_trian_datas.extend(datas)
        all_trian_labels.extend(labels)

    return np.array(all_trian_datas,dtype=np.float64), np.array(all_trian_labels,dtype=np.float64)

def load_CIFAR10(cifar10_dir):
    fo = open(os.path.join(cifar10_dir,'data_batch_1'))
    dict = cPickle.load(fo)
    fo.close()
    # train image datas shape is : (10000L, 3072L)
    datas = dict['data']
    # image labels :(10000L,)
    labels = dict['labels']

    return np.array(datas, dtype=np.float64), np.array(labels, dtype=np.float64)

def get_CIFAR10_data(num_training=9000, num_validation=1000, num_test=1000):
    """
        Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
        it for the two-layer neural net classifier. These are the same steps as
        we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    print 'loading cifar10 data.................'
    cifar10_dir = '/home/wxp/cifar10/cifar-10-batches-py'  # make a change
    X_train, y_train = load_CIFAR10(cifar10_dir)
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]  # (1000,3072)
    y_val = y_train[mask]  # (1000L,)
    mask = range(num_training)
    X_train = X_train[mask]  # (9000,3072)
    y_train = y_train[mask]  # (9000L,)
    mask = range(num_test)
    X_test = X_train[mask]  # (1000,3072)
    y_test = y_train[mask]  # (1000L,)

    # preprocessing: subtract the mean image
    mean_image = np.mean(X_train, axis=0,dtype=np.float64)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = np.array(X_train,dtype=np.float64).reshape(num_training, 3,32,32) # (49000, 3,32,32)
    X_val = np.array(X_val,dtype=np.float64).reshape(num_validation, 3,32,32) # (1000, 3,32,32)
    X_test = np.array(X_test,dtype=np.float64).reshape(num_test, 3,32,32) # (1000, 3,32,32)

    print 'loading cifar10 data over!!!!!!!!!!!!!!!!!!!'

    return X_train, y_train, X_val, y_val, X_test, y_test



# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train data dtype: ', X_train.dtype
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Look for the best net
best_net = None  # store the best model into this
net = softmax_test.softmaxTest()
"""
max_count = 100
for count in xrange(1, max_count + 1):
    reg = 10 ** np.random.uniform(-4, 1)
    lr = 10 ** np.random.uniform(-5, -3)
    stats = net.train(X_train, y_train, X_val, y_val, num_epochs=5, batch_size=200, mu=0.5, mu_increase=1.0, learning_rate=lr, learning_rate_decay=0.95, reg=reg, verbose=True)
    print 'val_acc: %f, lr: %s, reg: %s, (%d / %d)' % (stats['val_acc_history'][-1], format(lr, 'e'), format(reg, 'e'), count, max_count)

# according to the above experiment, reg ~= 0.9, lr ~= 5e-4
"""
stats = net.train(X_train, y_train, X_val, y_val, num_epochs=40, batch_size=400, mu=0.5, mu_increase=1.0,
                  learning_rate=5e-4, learning_rate_decay=0.95, reg=0.9, verbose=True)
# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print 'Validation accuracy: ', val_acc # about 52.7%

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.ylim([0, 0.8])
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend(bbox_to_anchor=(1.0, 0.4))
plt.grid(True)
plt.show()

# Run on the test set
test_acc = (net.predict(X_test) == y_test).mean()
print 'Test accuracy: ', test_acc # about 54.6%

