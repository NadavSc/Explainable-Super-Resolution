import pdb
import os
from PIL import Image
from torch.utils.data import DataLoader

from dataset import *
from logger.logger import info
from modules.TRAD import bicubic_interpolation, frequency_domain_sr_color


def test_bicubic(args):
    validation_dataset = DataExtractor(mode='validation',
                                        lr_path=args.db_valid_lr_path,
                                        hr_path=args.db_valid_hr_path,
                                        transform=None,
                                        eval=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    save_dir = './modules/TRAD/results/BICUBIC'
    os.makedirs(save_dir, exist_ok=True)

    for i, data in enumerate(validation_loader):
        lr_img = data['lr'].permute(2, 3, 1, 0)[:, :, :, 0].numpy()
        hr_img = data['hr'].permute(2, 3, 1, 0)[:, :, :, 0].numpy()
        
        lr_img = Image.fromarray(lr_img)
        hr_img = Image.fromarray(hr_img)

        sr_img = bicubic_interpolation(lr_img, hr_img.size)
        sr_img.save(os.path.join(save_dir, f'res_%04d.png' % i))
        info(f'Image {i} SR BICUBIC has been saved')


def test_fd(args):
    validation_dataset = DataExtractor(mode='validation',
                                    lr_path=args.db_valid_lr_path,
                                    hr_path=args.db_valid_hr_path,
                                    transform=None,
                                    eval=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    save_dir = './modules/TRAD/results/FD'
    os.makedirs(save_dir, exist_ok=True)


    for i, data in enumerate(validation_loader):
        lr_img = data['lr'].permute(2, 3, 1, 0)[:, :, :, 0].numpy()
        hr_img = data['hr'].permute(2, 3, 1, 0)[:, :, :, 0].numpy()
        
        
        lr_img = Image.fromarray(lr_img)
        hr_img = Image.fromarray(hr_img)

        sr_img = frequency_domain_sr_color(lr_img, scale=args.scale)
        sr_img.save(os.path.join(save_dir, f'res_%04d.png' % i))
        info(f'Image {i} SR FD has been saved')