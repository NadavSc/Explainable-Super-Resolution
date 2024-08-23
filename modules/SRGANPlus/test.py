import os
import pdb
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio

from dataset import *
from evaluate import plot_loss
from logger.logger import info
from modules.SRGANPlus import Generator, Discriminator, vgg19, TVLoss, perceptual_loss


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

    def std(outputs, mode='hsv'):
        # if mode == 'rgb':
        #     imgs = np.array([(np.transpose(output, (1, 2, 0)) * 255.0).astype(np.uint8) for output in outputs])
        # if mode == 'hsv':
        #     imgs = np.array([rgb_to_hsv((np.transpose(output, (1, 2, 0)) * 255.0).astype(np.uint8)) for output in outputs])
        if mode == 'rgb':
            imgs = np.array([np.transpose(output, (1, 2, 0)) for output in outputs])
        if mode == 'hsv':
            imgs = np.array([rgb_to_hsv(np.transpose(output, (1, 2, 0))) for output in outputs])


        std_dev = np.std(imgs, axis=0)
        std_dev_mean = np.mean(std_dev, axis=2)
        std_dev_normalized = (std_dev_mean - np.min(std_dev_mean)) / (np.max(std_dev_mean) - np.min(std_dev_mean))
        plt.figure(figsize=(10, 10))
        plt.imshow(std_dev_normalized, cmap='coolwarm', interpolation='nearest')
        plt.axis('off')
        plt.margins(0)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, 'STD', f'res_%04d_test.png' % i), pad_inches=0,  bbox_inches='tight', dpi=250)

    def generate(lr):
        output, _ = generator(lr)
        output = output[0].cpu().numpy()
        output = np.clip(output, -1.0, 1.0)
        output = (output + 1.0) / 2.0
        return output

    save_dir = f'./modules/SRGANPlus/results/SRGANPlus_{args.load_epoch}' if args.model_type == 'srgan'\
         else f'./modules/SRGANPlus/results/SRRESNETPlus_{args.load_epoch}'
    if args.bnn:
        save_dir += '_BNN'
        os.makedirs(os.path.join(save_dir, 'STD'), exist_ok=True)
        info('BNN is activated')
    else:
        generator.eval()

    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, 'result.txt'), 'w')
    psnr_list = []

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
                output = np.mean(outputs, axis=0)
                std(outputs)
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


def test_loss(args):
    validation_dataset = DataExtractor(mode='validation',
                                        lr_path=args.db_valid_lr_path,
                                        hr_path=args.db_valid_hr_path,
                                        transform=None)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    save_dir = f'./modules/SRGANPlus/results/SRGANPlus_{args.load_epoch}' if args.model_type == 'srgan'\
         else f'./modules/SRGANPlus/results/SRRESNETPlus_{args.load_epoch}'
    if args.bnn:
        save_dir += '_BNN'

    os.makedirs(save_dir, exist_ok=True)
    model_dir = fr'./modules/SRGANPlus/models/{args.model_type.upper()}Plus'
    models_paths = glob.glob(os.path.join(model_dir, '*.pt'))

    l2_loss = nn.MSELoss()
    loss_hist = []

    def generate(lr):
        output, _ = generator(lr)
        return output

    for model_path in models_paths:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator = generator.to(device)
        if not args.bnn:
            generator.eval()

        iter_loss = []
        with torch.no_grad():
            for i, te_data in enumerate(validation_loader):
                hr = te_data['hr'].type(torch.cuda.FloatTensor).to(device)
                lr = te_data['lr'].type(torch.cuda.FloatTensor).to(device)

                if args.bnn:
                    outputs = torch.stack([generate(lr) for i in range(100)], axis=0)
                    output = torch.mean(outputs, axis=0)
                else:
                    output = generate(lr)

                loss = l2_loss(hr, output)

                iter_loss.append(loss.item())

            loss_hist.append(np.average(iter_loss))
            info(f'{os.path.basename(model_path)} Loss: {loss_hist[-1]:.07f}')
    np.save(os.path.join(save_dir, 'test_loss.npy'), np.array(loss_hist))
    plot_loss(args)
    info('train and test loss plot has been saved')