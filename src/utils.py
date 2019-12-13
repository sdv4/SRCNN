"""
   Functions of general utility that are used throughout the project.
"""

import numpy as np
import torch
import torch.nn as nn

def ycbcr2rgb(im):
    """
        Takes images in YCbCr format and converts it to RGB

        Args:
            im (np.ndarray): image in YCbCr format
        Returns:
            rgb (np.ndarray): im in rgb format
        source: https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb
    """
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def psnr(img1, img2):
    """
        Calculate the Peak Signal to Noise Ratio between two images.
        formula: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img1 (np.ndarray): image of any number of channels
            img2 (np.ndarray): image of any number of channels

        Returns:
            psnr (float): psnr between img1 and img2 in dB
    """
    mse = np.mean(np.power(img1.astype(np.double) - img2.astype(np.double), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    p_snr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return p_snr
