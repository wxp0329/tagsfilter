# encoding=utf-8
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_new = 1 + (H + 2 * pad - HH) / stride
    W_new = 1 + (W + 2 * pad - WW) / stride
    s = stride
    out = np.zeros((N, F, H_new, W_new))

    for i in xrange(N):  # ith image
        for f in xrange(F):  # fth filter
            for j in xrange(H_new):
                for k in xrange(W_new):

                    out[i, f, j, k] = np.sum(x_padded[i, :, j * s:HH + j * s, k * s:WW + k * s] * w[f]) + b[f]

    cache = (x, w, b, conv_param)

    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + (H + 2 * pad - HH) / stride
    W_new = 1 + (W + 2 * pad - WW) / stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in xrange(N):  # ith image
        for f in xrange(F):  # fth filter
            for j in xrange(H_new):
                for k in xrange(W_new):
                    window = x_padded[i, :, j * s:HH + j * s, k * s:WW + k * s]
                    db[f] += dout[i, f, j, k]
                    dw[f] += window * dout[i, f, j, k]
                    dx_padded[i, :, j * s:HH + j * s, k * s:WW + k * s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad + H, pad:pad + W]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) / s
    W_new = 1 + (W - WW) / s
    out = np.zeros((N, C, H_new, W_new))

    for i in xrange(N):
        for j in xrange(C):
            for k in xrange(H_new):
                for l in xrange(W_new):
                    window = x[i, j, k * s:HH + k * s, l * s:WW + l * s]
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) / s
    W_new = 1 + (W - WW) / s
    dx = np.zeros_like(x)

    for i in xrange(N):
        for j in xrange(C):
            for k in xrange(H_new):
                for l in xrange(W_new):
                    window = x[i, j, k * s:HH + k * s, l * s:WW + l * s]
                    m = np.max(window)
                    dx[i, j, k * s:HH + k * s, l * s:WW + l * s] = (window == m) * dout[i, j, k, l]

    return dx


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)
    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    # Reshape x into rows
    N = x.shape[0]
    x_row = x.reshape(N, -1)  # (N,D)
    out = np.dot(x_row, w) + b  # (N,M)
    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dx = np.dot(dout, w.T)  # (N,D)
    dx = np.reshape(dx, x.shape)  # (N,d1,...,d_k)
    x_row = x.reshape(x.shape[0], -1)  # (N,D)
    dw = np.dot(x_row.T, dout)  # (D,M)
    db = np.sum(dout, axis=0, keepdims=True)  # (1,M)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = ReLU(x)
    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x <= 0] = 0

    return dx


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    out, cache_conv = conv_forward_naive(x, w, b, conv_param)
    out1, cache_relu = relu_forward(out)
    out2, cache_pool = max_pool_forward_naive(out1, pool_param)
    cache = (cache_conv, cache_relu, cache_pool)
    return out2, cache


def conv_relu_pool_backward(dout, cache):
    cache_conv, cache_relu, cache_pool = cache
    dx = max_pool_backward_naive(dout, cache_pool)
    dx = relu_backward(dx, cache_relu)
    dx, dw, db = conv_backward_naive(dx, cache_conv)

    return dx, dw, db


def affine_relu_forward(x, w, b):
    out, cache_affine = affine_forward(x, w, b)
    out1, cache_relu = relu_forward(out)
    cache = (cache_affine, cache_relu)

    return out1, cache


def affine_relu_backward(dout, cache):
    cache_affine, cache_relu = cache
    dx = relu_backward(dout, cache_relu)
    dx, dw, db = affine_backward(dx, cache_affine)

    return dx, dw, db


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
         for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
         0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx


def ReLU(x):
    """ReLU non-linearity."""
    return np.maximum(0, x)


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0, keepdims=True)  # [1,D]
        sample_var = np.var(x, axis=0, keepdims=True)  # [1,D]
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)  # [N,D]
        out = gamma * x_normalized + beta
        cache = (x_normalized, gamma, beta, sample_mean, sample_var, x, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'test':
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    x_normalized, gamma, beta, sample_mean, sample_var, x, eps = cache
    N, D = x.shape
    dx_normalized = dout * gamma  # [N,D]
    x_mu = x - sample_mean  # [N,D]
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)  # [1,D]
    dsample_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0, keepdims=True) * sample_std_inv ** 3
    dsample_mean = -1.0 * np.sum(np.array(dx_normalized * sample_std_inv), axis=0, keepdims=True) - \
                   2.0 * dsample_var * np.mean(x_mu, axis=0, keepdims=True)

    dx1 = dx_normalized * sample_std_inv
    dx2 = 2.0 / N * dsample_var * x_mu
    dx = dx1 + dx2 + 1.0 / N * dsample_mean
    dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    N, C, H, W = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta
