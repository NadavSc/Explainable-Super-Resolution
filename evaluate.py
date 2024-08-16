import os
import pdb
import torch
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2ycbcr

from dataset import DataExtractor
from torch.utils.data import DataLoader
from logger.logger import info


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def calculate_lpips(lpips_model, img1, img2):
    tA = t(img1[:,:,[2,1,0]]).to(torch.cuda.current_device())
    tB = t(img2[:,:,[2,1,0]]).to(torch.cuda.current_device())
    return lpips_model.forward(tA, tB).item()


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = rgb2ycbcr(img1)  # ycbcr is more common in PSNR calculations
    img2 = rgb2ycbcr(img2)
    if img1.shape[-1] == 3:
        img1 = img1[:, :, :1]
        img2 = img2[:, :, :1]
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def evaluate(args):
    validation_dataset = DataExtractor(mode='validation',
                                            lr_path=args.db_valid_sr_path,
                                            hr_path=args.db_valid_hr_path,
                                            transform=None,
                                            eval=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    import lpips
    lpips_model = lpips.LPIPS(net='alex')
    lpips_model.to(torch.cuda.current_device())

    f = open(os.path.join(args.db_valid_sr_path, 'results.txt'), 'w')
    for i, data in enumerate(validation_loader):
        sr_img = data['lr']
        hr_img = data['hr']

        sr_img = sr_img[0, :, :, :].permute(1, 2, 0)
        hr_img = hr_img[0, :, :, :].permute(1, 2, 0)

        sr_img = np.array(sr_img)
        hr_img = np.array(hr_img)
        lpips = calculate_lpips(lpips_model, sr_img, hr_img)
        psnr = calculate_psnr(sr_img, hr_img)
        ssim_value = ssim(sr_img, hr_img, win_size=sr_img.shape[-1], channel_axis=-1)
        result = f'Image {i} PSNR: {psnr:.04f} | SSIM: {ssim_value:.04f} | LPIPS: {lpips:.04f}'
        info(result)
        f.write(result+'\n')
    f.close()