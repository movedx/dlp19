from __future__ import print_function
from builtins import range
import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    assert conv_param['stride'] > 0
    assert x.shape[1] == w.shape[1]
    assert len(x.shape) == 4

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    x_pad = np.pad(x, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant',
                   constant_values=0)

    out_w = 1 + (H + 2 * pad - HH) // stride
    out_h = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros(shape=(N, F, out_w, out_h))

    for sample_idx, sample in enumerate(x_pad):
        for kernel_idx, kernel in enumerate(w):
            for i in range(out_h):
                for j in range(out_w):
                    out[sample_idx][kernel_idx][i][j] = np.sum(
                        kernel * sample[
                                 :, i * stride:i * stride + HH, j * stride:j * stride + WW
                                 ]) + b[kernel_idx]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, dout_h, dout_w = dout.shape

    pad = conv_param['pad']
    stride = conv_param['stride']

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    dw = np.zeros((F, C, HH, WW))

    for f in range(F):
        for c in range(C):
            for i in range(HH):
                for j in range(WW):
                    sub_pad = x_pad[:, c, i:i + dout_h * stride:stride, j:j + dout_w * stride:stride]
                    dw[f, c, i, j] = np.sum(
                        dout[:, f, :, :] * sub_pad)

    db = np.zeros(F)

    for f in range(F):
        db[f] = np.sum(dout[:, f, :, :])

    dx = np.zeros((N, C, H, W))

    for n in range(N):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(dout_h):
                        for l in range(dout_w):
                            mask1 = np.zeros_like(w[f])
                            mask2 = np.zeros_like(w[f])
                            if HH > (i + pad - k * stride) >= 0:
                                mask1[:, i + pad - k * stride, :] = 1.0
                            if WW > (j + pad - l * stride) >= 0:
                                mask2[:, :, j + pad - l * stride] = 1.0
                            w_masked = np.sum(w[f] * mask1 * mask2, axis=(1, 2))
                            dx[n, :, i, j] += dout[n, f, k, l] * w_masked

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################

    N, C, H, W = x.shape

    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    out_h = (H - pool_h) // stride + 1
    out_w = (W - pool_w) // stride + 1

    out = np.zeros((N, C, out_h, out_w))

    for n in range(N):
        for c in range(C):
            for k in range(out_h):
                for l in range(out_w):
                    out[n, c, k, l] = np.max(
                        x[n, c, k * stride:k * stride + pool_h, l * stride:l * stride + pool_w])

    cache = (x, pool_param)
    return out, cache
   
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################

    x, pool_param = cache

    N, C, H, W = x.shape

    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']

    h = (H - pool_h) // stride + 1
    w = (W - pool_w) // stride + 1

    dx = np.zeros((N, C, H, W))

    for nprime in range(N):
        for cprime in range(C):
            for k in range(h):
                for l in range(w):
                    x_pooling = x[nprime, cprime, k *
                                  stride:k * stride + pool_h, l * stride:l * stride + pool_w]
                    maxi = np.max(x_pooling)
                    x_mask = x_pooling == maxi
                    dx[nprime, cprime, k * stride:k * stride + pool_h, l * stride:l *
                        stride + pool_w] += dout[nprime, cprime, k, l] * x_mask
    return dx
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for dropout.            #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for dropout.            #
        #######################################################################

        mask = None
        out = x

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for dropout            #
        #######################################################################

        dx = dout * mask

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
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
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = np.dot(x.reshape((x.shape[0],-1)),w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.dot(dout,w.T).reshape(x.shape)
    dw = np.dot(x.reshape((x.shape[0],-1)).T,dout)
    db = np.sum(dout,axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
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
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx =  np.multiply(x>0,dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
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
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
