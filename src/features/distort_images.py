"""
    Functions used for distorting images for training
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import torch

from src.utils import ycbcr2rgb

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def distort_image(path, factor, sigma=1, blur=True):
    """ Distorts image by bluring it, decreasing its resolution
        by some factor, then increasing resolution - by bicubic
        interpolation.

        Args:
            path (string): absolute path to an image file
            factor (int): the resolution factor for interpolation
            sigma (float): the std. dev. to use for the gaussian blur
            blur (boolean): if True, gaussian blur is performed on im
        Returns:
            blurred_img (numpy.ndarray): distorted image in YCbCr with
                type uint8

    """
    image_file = Image.open(path)
    im = np.array(image_file.convert('YCbCr'))
    im_Y, im_Cb, im_Cr = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    im_Y = (im_Y.astype(np.int16)).astype(np.int64)
    im_Cb = (im_Cb.astype(np.int16)).astype(np.int64)
    im_Cr = (im_Cr.astype(np.int16)).astype(np.int64)
    if blur:
        im_Y_blurred = gaussian_filter(im_Y, sigma=sigma)
    else:
        im_Y_blurred = im_Y
    im_blurred = np.copy(im)
    im_blurred[:, :, 0] = im_Y_blurred
    im_blurred[:, :, 1] = im_Cb
    im_blurred[:, :, 2] = im_Cr
    width, length = im_Y.shape
    im_blurred = Image.fromarray(im_blurred, mode='YCbCr')
    im_blurred = im_blurred.resize(size=(int(length/factor),
                                         int(width/factor)),
                                   resample=Image.BICUBIC)

    im_blurred = im_blurred.resize(size=(length, width),
                                   resample=Image.BICUBIC)
    im_blurred = np.array(im_blurred.convert('YCbCr'))
    return im_blurred
