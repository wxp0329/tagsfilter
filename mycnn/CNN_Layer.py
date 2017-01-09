# coding:utf-8
import cPickle
import numpy as np
from PIL import Image
import layers
import optim
import NUS_loss_test


class CNNLayer(object):
    def __init__(self, input_dim=(3, 240, 240), num_filters=32, filter_size=3, hidden_dim=192,
                 weight_scale=1e-3, reg=0.0, dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize weights and biases
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros((num_filters))
        self.params['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, hidden_dim)
        self.params['b2'] = np.zeros((hidden_dim))
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
        self.params['b3'] = np.zeros((hidden_dim))

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None, reg=1e-5):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # compute the forward pass
        print 'compute the conv_relu_pool_forward forward pass'
        a1, cache1 = layers.conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        norm_out, norm_cache = layers.spatial_batchnorm_forward(a1, 1, 0, bn_param={'mode': 'train'})

        print 'compute the affine_relu_forward forward pass'
        a2, cache2 = layers.affine_relu_forward(norm_out, W2, b2)
        scores, cache3 = layers.affine_forward(a2, W3, b3)

        if y is None:
            return scores

        # compute the backward pass
        print 'compute the NUS_loss backward pass'
        data_loss = NUS_loss_test.NUSDataTrain().loss(scores, y)
        dscores = NUS_loss_test.NUSDataTrain().eval_numerical_gradient(NUS_loss_test.NUSDataTrain().grad_loss, scores)  # layers.softmax_loss(scores, y)#改这里
        da2, dW3, db3 = layers.affine_backward(dscores, cache3)
        print 'compute the affine_relu_backward backward pass'
        da1, dW2, db2 = layers.affine_relu_backward(da2, cache2)
        print 'compute the spatial_batchnorm_backward backward pass'
        dnorm_out, dgamma, dbeta = layers.spatial_batchnorm_backward(da1, norm_cache)
        print 'compute the conv_relu_pool_backward backward pass'
        dX, dW1, db1 = layers.conv_relu_pool_backward(dnorm_out, cache1)

        # Add regularization
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])
        loss = data_loss + reg_loss
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

        return loss, grads

    def NUStrain(self, X, y, learning_rate=1e-3,
                 learning_rate_decay=0.95, reg=1e-5, mu=0.9, num_epochs=10,
                 mu_increase=1.0, batch_size=200, verbose=False):
        """
                Train this neural network using stochastic gradient descent.
                Inputs:
                - X: A numpy array of shape (N, D) giving training data.
                - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
                     X[i] has label c, where 0 <= c < C.
                - X_val: A numpy array of shape (N_val, D) giving validation data.
                - y_val: A numpy array of shape (N_val,) giving validation labels.
                - learning_rate: Scalar giving learning rate for optimization.
                - learning_rate_decay: Scalar giving factor used to decay the learning rate
                                       after each epoch.
                - reg: Scalar giving regularization strength.
                - num_iters: Number of steps to take when optimizing.
                - batch_size: Number of training examples to use per step.
                - verbose: boolean; if true print progress during optimization.
                """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        # Use SGD to optimize the parameters
        v_W3, v_b3 = 0.0, 0.0
        v_W2, v_b2 = 0.0, 0.0
        v_W1, v_b1 = 0.0, 0.0
        config_vW3 = {}
        config_vb3 = {}
        config_vW2 = {}
        config_vb2 = {}
        config_vW1 = {}
        config_vb1 = {}

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in xrange(1, num_epochs * iterations_per_epoch + 1):
            X_batch = None
            y_batch = None

            # Sampling with replacement is faster than sampling without replacement.
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index, :]  # (batch_size,D)
            y_batch = y[sample_index]  # (1,batch_size)

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # apply decay learning_rate,increase momentum
            config_vW3['momentum'] = mu
            config_vW3['learning_rate'] = learning_rate
            config_vb3['momentum'] = mu
            config_vb3['learning_rate'] = learning_rate

            config_vW2['momentum'] = mu
            config_vW2['learning_rate'] = learning_rate
            config_vb2['momentum'] = mu
            config_vb2['learning_rate'] = learning_rate

            config_vW1['momentum'] = mu
            config_vW1['learning_rate'] = learning_rate
            config_vb1['momentum'] = mu
            config_vb1['learning_rate'] = learning_rate

            # Perform parameter update (with momentum)
            print ' Perform parameter update (with momentum)'
            v_W3, config_vW3 = optim.sgd_momentum(self.params['W3'], grads['W3'], config_vW3)
            self.params['W3'] = v_W3
            v_b3, config_vb3 = optim.sgd_momentum(self.params['b3'], grads['b3'], config_vb3)
            self.params['b3'] = v_b3

            v_W2, config_vW2 = optim.sgd_momentum(self.params['W2'], grads['W2'], config_vW2)
            self.params['W2'] = v_W2
            v_b2, config_vb2 = optim.sgd_momentum(self.params['b2'], grads['b2'], config_vb2)
            self.params['b2'] = v_b2

            v_W1, config_vW1 = optim.sgd_momentum(self.params['W1'], grads['W1'], config_vW1)
            self.params['W1'] = v_W1
            v_b1, config_vb1 = optim.sgd_momentum(self.params['b1'], grads['b1'], config_vb1)
            self.params['b1'] = v_b1

            # print loss
            epoch = it / iterations_per_epoch
            print
            print 'epoch %d / %d: loss is %f' % (
                epoch, num_epochs, loss)
            # Every epoch, check decay learning rate.
            if verbose and it % iterations_per_epoch == 0:
                # Decay learning rate
                learning_rate *= learning_rate_decay
                # Increase mu
                mu *= mu_increase
                print 'write model into /home/wxp/mytest.model file!!!!!!!!!!!!!'
                with open('/home/wxp/' + it + '_mytest.model', 'w') as fw:
                    cPickle.dump(self.params, fw)

        return {'loss_history': loss_history}

    def train(self, X, y, X_val, y_val, learning_rate=1e-3,
              learning_rate_decay=0.95, reg=1e-5, mu=0.9, num_epochs=10,
              mu_increase=1.0, batch_size=200, verbose=False):
        """
                Train this neural network using stochastic gradient descent.
                Inputs:
                - X: A numpy array of shape (N, D) giving training data.
                - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
                     X[i] has label c, where 0 <= c < C.
                - X_val: A numpy array of shape (N_val, D) giving validation data.
                - y_val: A numpy array of shape (N_val,) giving validation labels.
                - learning_rate: Scalar giving learning rate for optimization.
                - learning_rate_decay: Scalar giving factor used to decay the learning rate
                                       after each epoch.
                - reg: Scalar giving regularization strength.
                - num_iters: Number of steps to take when optimizing.
                - batch_size: Number of training examples to use per step.
                - verbose: boolean; if true print progress during optimization.
                """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        # Use SGD to optimize the parameters
        v_W3, v_b3 = 0.0, 0.0
        v_W2, v_b2 = 0.0, 0.0
        v_W1, v_b1 = 0.0, 0.0
        config_vW3 = {}
        config_vb3 = {}
        config_vW2 = {}
        config_vb2 = {}
        config_vW1 = {}
        config_vb1 = {}

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        for it in xrange(1, num_epochs * iterations_per_epoch + 1):
            X_batch = None
            y_batch = None

            # Sampling with replacement is faster than sampling without replacement.
            sample_index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[sample_index, :]  # (batch_size,D)
            y_batch = y[sample_index]  # (1,batch_size)

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # apply decay learning_rate,increase momentum
            config_vW3['momentum'] = mu
            config_vW3['learning_rate'] = learning_rate
            config_vb3['momentum'] = mu
            config_vb3['learning_rate'] = learning_rate

            config_vW2['momentum'] = mu
            config_vW2['learning_rate'] = learning_rate
            config_vb2['momentum'] = mu
            config_vb2['learning_rate'] = learning_rate

            config_vW1['momentum'] = mu
            config_vW1['learning_rate'] = learning_rate
            config_vb1['momentum'] = mu
            config_vb1['learning_rate'] = learning_rate

            # Perform parameter update (with momentum)
            v_W3, config_vW3 = optim.sgd_momentum(self.params['W3'], grads['W3'], config_vW3)
            self.params['W3'] = v_W3
            v_b3, config_vb3 = optim.sgd_momentum(self.params['b3'], grads['b3'], config_vb3)
            self.params['b3'] = v_b3

            v_W2, config_vW2 = optim.sgd_momentum(self.params['W2'], grads['W2'], config_vW2)
            self.params['W2'] = v_W2
            v_b2, config_vb2 = optim.sgd_momentum(self.params['b2'], grads['b2'], config_vb2)
            self.params['b2'] = v_b2

            v_W1, config_vW1 = optim.sgd_momentum(self.params['W1'], grads['W1'], config_vW1)
            self.params['W1'] = v_W1
            v_b1, config_vb1 = optim.sgd_momentum(self.params['b1'], grads['b1'], config_vb1)
            self.params['b1'] = v_b1

            """
                 if verbose and it % 100 == 0:
                 print 'iteration %d / %d: loss %f' % (it, num_iters, loss)
            """
            # Every epoch, check train and val accuracy and decay learning rate.
            if verbose and it % iterations_per_epoch == 0:
                # Check accuracy
                epoch = it / iterations_per_epoch
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                print 'epoch %d / %d: loss %f, train_acc: %f, val_acc: %f' % (
                    epoch, num_epochs, loss, train_acc, val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay
                # Increase mu
                mu *= mu_increase
        return {'loss_history': loss_history, 'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history, }

    def predict(self, X):
        """
            Inputs:
                - X: A numpy array of shape (N, D) giving N D-dimensional data points to
                     classify.
                Returns:
                - y_pred: A numpy array of shape (N,) giving predicted labels for each of
                          the elements of X. For all i, y_pred[i] = c means that X[i] is
                          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None
        h1 = layers.ReLU(np.dot(X, self.params['W1']) + self.params['b1'])
        scores = np.dot(h1, self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        return y_pred
