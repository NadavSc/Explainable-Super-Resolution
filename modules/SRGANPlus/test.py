import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio

from dataset import *
from logger.logger import info
from modules.SRGANPlus import Generator, Discriminator, vgg19, TVLoss, perceptual_loss


def combine_predictions(matrices):
    """
    Computes the average per pixel of the given list of matrices.
    
    Parameters:
    matrices: List of 2D numpy arrays (matrices) of the same size.
    
    Returns:
    A 2D numpy array representing the average matrix.
    """
    if len(matrices) == 0:
        raise ValueError("The list of matrices is empty.")
    
    # Check if all matrices have the same shape
    if not all(mat.shape == matrices[0].shape for mat in matrices):
        raise ValueError("All matrices must have the same dimensions.")
    
    # Stack the matrices along a new axis
    stacked_matrices = np.stack(matrices, axis=0)
    
    # Compute the average across the new axis
    average_img = np.mean(stacked_matrices, axis=0)
    std_img = np.std(stacked_matrices, axis=0)

    return average_img, std_img


def test(args):
    validation_dataset = DataExtractor(mode='validation',
                                        lr_path=args.db_valid_lr_path,
                                        hr_path=args.db_valid_hr_path,
                                        transform=None)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    generator = generator.to(device)
    generator.eval()

    save_dir = f'./modules/SRGANPlus/results/SRGANPlus_{args.load_epoch}' if args.model_type == 'srgan'\
         else f'./modules/SRGANPlus/results/SRRESNETPlus_{args.load_epoch}'
    if args.bnn:
        save_dir += '_BNN'
        os.makedirs(os.path.join(save_dir, 'STD'), exist_ok=True)
        info('BNN is activated')

    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, 'result.txt'), 'w')
    psnr_list = []

    def generate(lr):
        output, _ = generator(lr)
        output = output[0].cpu().numpy()
        output = np.clip(output, -1.0, 1.0)
        output = (output + 1.0) / 2.0
        return output

    with torch.no_grad():
        for i, te_data in enumerate(validation_loader):
            hr = te_data['hr'].type(torch.cuda.FloatTensor).to(device)
            lr = te_data['lr'].type(torch.cuda.FloatTensor).to(device)

            bs, c, h, w = lr.size()
            hr = hr[:, :, : h * args.scale, : w * args.scale]
            hr = hr[0].cpu().numpy()
            hr = (hr + 1.0) / 2.0

            if args.bnn:
                outputs = np.array([generate(lr) for i in range(100)])
                import pdb; pdb.set_trace()
                output, std_img = combine_predictions(outputs)
                #np.savez(os.path.join(save_dir, 'STD/res_std_%04d.npy' % i), std_img)
                std_img = std_img.transpose(1, 2, 0)
                std_img = (std_img - std_img.min()) / (std_img.max() - std_img.min())
                std_img = (std_img * 255)
                plt.imshow(std_img)
                plt.axis('off')
                plt.margins(0)
                plt.savefig(os.path.join(save_dir, f'STD/res_std_img_%04d.png' % i), bbox_inches='tight', pad_inches=0, dpi=200)
                plt.close()
            else:
                output = generate(lr)

            output = output.transpose(1, 2, 0)
            hr = hr.transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_hr = rgb2ycbcr(hr)[args.scale:-args.scale, args.scale:-args.scale, :1]

            psnr = peak_signal_noise_ratio(y_output / 255.0, y_hr / 255.0, data_range=1.0)
            psnr_list.append(psnr)
            info(f'Image {i} SRGAN PSNR: {psnr:.04f}')
            f.write('psnr : %04f \n' % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save(os.path.join(save_dir, f'res_%04d.png' % i))
        info(f'Average PSNR: {np.mean(psnr_list):.04f}')
        f.write('avg psnr : %04f' % np.mean(psnr_list))