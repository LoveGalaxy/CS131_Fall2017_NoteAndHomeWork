import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * (image ** 2) / 255
    ### END YOUR CODE
    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    #t = np.sum(image * np.array([1, 1, 1]) / 3, axis=2)
    (h, w, x) = image.shape
    out = np.zeros((h, w, x))

#    p = [0.2989, 0.5870, 0.1140]
    p = [0.3333, 0.3333, 0.3333]
    t = p[0] * image[:, :, 0] + p[1] * image[:, :, 1] + p[2] * image[:, :, 2]
    for i in range(3):
       out[:,:,i] = t / 255
    ### END YOUR CODE
    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    (h, w, x) = image.shape
    out = np.zeros((h, w, x))
    if channel == 'R':
        out[:, :, 1] = image[:, :, 1]
        out[:, :, 2] = image[:, :, 2]
    elif channel == 'G':
        out[:, :, 0] = image[:, :, 0]
        out[:, :, 2] = image[:, :, 2]
    else:
        out[:, :, 0] = image[:, :, 0]
        out[:, :, 1] = image[:, :, 1]
    out = out / 255
    ### END YOUR CODE
    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE

    (h, w, x) = lab.shape
    out = np.zeros((h, w, x))
    if channel == 'L':
        out[:, :, 1] = lab[:, :, 1]
        out[:, :, 2] = lab[:, :, 2]
    elif channel == 'A':
        out[:, :, 0] = lab[:, :, 0]
        out[:, :, 2] = lab[:, :, 2]
    else:
        out[:, :, 0] = lab[:, :, 0]
        out[:, :, 1] = lab[:, :, 1]

    out = color.lab2rgb(out)
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    (h, w, x) = hsv.shape
    out = np.zeros((h, w, x))
    if channel == 'H':
        out[:, :, 1] = hsv[:, :, 1]
        out[:, :, 2] = hsv[:, :, 2]
    elif channel == 'S':
        out[:, :, 0] = hsv[:, :, 0]
        out[:, :, 2] = hsv[:, :, 2]
    else:
        out[:, :, 0] = hsv[:, :, 0]
        out[:, :, 1] = hsv[:, :, 1] 

    out = color.hsv2rgb(out)
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    t1 = rgb_decomposition(image1, channel1) / 2
    t2 = rgb_decomposition(image2, channel2) / 2
    out = t1 + t2
    ### END YOUR CODE

    return out
