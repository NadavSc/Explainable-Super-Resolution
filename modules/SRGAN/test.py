import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio

from dataset import *
from logger.logger import info
from modules.SRGAN import Generator, Discriminator, vgg19, TVLoss, perceptual_loss


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

    save_dir = './modules/SRGAN/results/SRGAN' if args.model_type == 'srgan'\
         else './modules/SRGAN/results/SRRESNET'
    f = open(os.path.join(save_dir, 'result.txt'), 'w')
    psnr_list = []

    with torch.no_grad():
        for i, te_data in enumerate(validation_loader):
            hr = te_data['hr'].type(torch.cuda.FloatTensor).to(device)
            lr = te_data['lr'].type(torch.cuda.FloatTensor).to(device)

            bs, c, h, w = lr.size()
            hr = hr[:, :, : h * args.scale, : w * args.scale]

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            hr = hr[0].cpu().numpy()

            output = (output + 1.0) / 2.0
            hr = (hr + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            hr = hr.transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
            y_hr = rgb2ycbcr(hr)[args.scale:-args.scale, args.scale:-args.scale, :1]

            psnr = peak_signal_noise_ratio(y_output / 255.0, y_hr / 255.0, data_range=1.0)
            psnr_list.append(psnr)
            info(f'Image {i} {args.model_type.upper} PSNR: {psnr:.04f}')
            f.write('psnr : %04f \n' % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save(os.path.join(save_dir, f'res_%04d.png' % i))
        info(f'Average PSNR: {np.mean(psnr_list):.04f}')
        f.write('avg psnr : %04f' % np.mean(psnr_list))


def predict(args, img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    i=100
    with torch.no_grad():
        lr = torch.tensor(img).to(device)
        bs, c, h, w = lr.size()

        output, _ = generator(lr)

        output = output[0].cpu().numpy()
        output = np.clip(output, -1.0, 1.0)

        output = (output + 1.0) / 2.0
        output = output.transpose(1, 2, 0)

        y_output = rgb2ycbcr(output)[args.scale:-args.scale, args.scale:-args.scale, :1]
        result = Image.fromarray((output * 255.0).astype(np.uint8))
        result.save(f'./modules/SRGAN/results/res_%04d.png' % i)
