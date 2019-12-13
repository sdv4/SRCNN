"""
    Functions for evaluating model performance.
"""
import os

import numpy as np
from PIL import Image
import torch

from src.features.distort_images import distort_image
from src.utils import psnr

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def evaluate_model(path, model, pixel_mean, pixel_std, SR_FACTOR=3, sigma=1):
    """
        Computes average Peak Signal to Noise Ratio over a set of images and
        the versions super resolved by a srcnn or variant.

        Args:
            path (string): relative path to directory containing images for evaluation
            model (PyTorch model): the model to be evaluated
            pixel_mean (float): mean luminance value to be used for standardization
            pixel_std (float): std. dev. of luminance value to be used for standardization
            SR_FACTOR (int): super resolution factor
            sigma (int): the std. dev. to use for the gaussian blur
    """

    img_names = [im for im in os.listdir(path) if im[-4:] == '.bmp' or im[-4:] == '.jpg']
    blurred_img_psnrs = []
    out_img_psnrs = []
    for test_im in img_names:

        blurred_test_im = distort_image(path=path+test_im, factor=SR_FACTOR, sigma=sigma)
        ImageFile = Image.open(path+test_im)
        im = np.array(ImageFile.convert('YCbCr'))

        #normalize
        model_input = blurred_test_im[:, :, 0] / 255.0
        #standardize
        model_input -= pixel_mean
        model_input /= pixel_std

        im_out_Y = model(torch.tensor(model_input,
                                      dtype=torch.float).unsqueeze(0).unsqueeze(0).to(DEVICE))
        im_out_Y = im_out_Y.detach().squeeze().squeeze().cpu().numpy().astype(np.float64)
        im_out_viz = np.zeros((im_out_Y.shape[0], im_out_Y.shape[1], 3))

        #unstandardize
        im_out_Y = (im_out_Y * pixel_std) + pixel_mean

        #un-normalize
        im_out_Y *= 255.0

        im_out_viz[:, :, 0] = im_out_Y
        im_out_viz[:, :, 1] = im[:, :, 1]
        im_out_viz[:, :, 2] = im[:, :, 2]

        im_out_viz[:, :, 0] = np.around(im_out_viz[:, :, 0])
        #psnr on Y only
        blur_psnr = psnr(im[:, :, 0], blurred_test_im[:, :, 0])
        sr_psnr = psnr(im[:, :, 0], im_out_viz[:, :, 0])

        blurred_img_psnrs.append(blur_psnr)
        out_img_psnrs.append(sr_psnr)

    mean_blur_psnr = np.mean(np.array(blurred_img_psnrs))
    mean_sr_psnr = np.mean(np.array(out_img_psnrs))
    return mean_blur_psnr, mean_sr_psnr
