import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def bicubic_interpolation(lr_img, hr_img_size):
    return lr_img.resize(hr_img_size, Image.BICUBIC)


def frequency_domain_sr_single_channel(lr_image_channel, scale):
    lr_image_channel = lr_image_channel / 255.0

    f_lr = np.fft.fft2(lr_image_channel)
    f_lr_shifted = np.fft.fftshift(f_lr)

    h, w = lr_image_channel.shape
    hr_h, hr_w = h * scale, w * scale

    pad_h = (hr_h - h) // 2
    pad_w = (hr_w - w) // 2

    f_hr_shifted = np.pad(f_lr_shifted,
                          ((pad_h, hr_h - h - pad_h),
                           (pad_w, hr_w - w - pad_w)),
                          mode='constant')

    f_hr = np.fft.ifftshift(f_hr_shifted)
    hr_image_channel = np.abs(np.fft.ifft2(f_hr))
    hr_image_channel = hr_image_channel / np.max(hr_image_channel) * 255.0
    hr_image_channel = np.clip(hr_image_channel, 0, 255).astype(np.uint8)

    return hr_image_channel


def frequency_domain_sr_color(lr_image, scale=4):

    lr_image_np = np.array(lr_image)
    r_channel, g_channel, b_channel = lr_image_np[:,:,0], lr_image_np[:,:,1], lr_image_np[:,:,2]

    r_hr = frequency_domain_sr_single_channel(r_channel, scale)
    g_hr = frequency_domain_sr_single_channel(g_channel, scale)
    b_hr = frequency_domain_sr_single_channel(b_channel, scale)

    hr_image_np = np.stack((r_hr, g_hr, b_hr), axis=-1)
    hr_image = Image.fromarray(hr_image_np)

    return hr_image