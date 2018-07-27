import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            sum = 0
            for i in range(Hk):
                for j in range(Wk):
                    if m+1-i < 0 or n+1-j < 0 or m+1-i >= Hi or n+1-j >= Wi:
                        sum += 0
                    else:
                        sum += kernel[i][j] * image[m+1-i][n+1-j]
            out[m][n] = sum   
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    for i in range(H):
        for j in range(W):
            out[i+pad_height][j+pad_width] = image[i][j]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    zero_pad_img = zero_pad(image, Hk//2, Wk//2)
    # flip kernel
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(zero_pad_img[i:i+Hk, j:j+Wk] * kernel)

    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # 两个思路
    # 1. 将卷积核和图像变到频域做乘积
    # 2. 像这个卷积核可以拆解成两个一维向量的乘积，做两个一维的卷积计算量小于做一个二维的卷积

    #1
    """   
    print(Hi, Wi)
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    f_kernel = zero_pad(kernel, 148, 164)
    f = np.fft.fft2(f_kernel)
    f = np.fft.fftshift(f)

    f_img = np.fft.fft2(image)
    f_img = np.fft.fftshift(f_img)
    #f * f_img
    out = f * f_img
    out = np.fft.ifftshift(out)
    out = np.fft.ifft2(out)
    out = np.abs(np.real(out))
    print(out)
    """

    # 2
    # 拆解部分还可以重写
    
    kernel_row = np.array([-1, 0, 1])
    kernel_col = np.array([[1], [2], [1]])
    Hp = Hk // 2
    Wp = Wk // 2
    zero_pad_img = zero_pad(image, Hp, Wp)

    for i in range(Hi):
        zero_pad_img[i,:] = np.sum(zero_pad_img[i:i+Hk,:] * kernel_col, 0)
    for i in range(Wi):
        out[:,i] = np.sum(zero_pad_img[:-2,i:i+Wk] * kernel_row, 1)
    
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE

    g = np.flip(g, 0)
    g = np.flip(g, 1)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g = g - np.mean(g)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    f = (f - np.mean(f)) / np.std(f)
    g = (g - np.mean(g)) / np.std(g)
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out
