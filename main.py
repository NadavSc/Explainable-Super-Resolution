import os
import argparse
import subprocess

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import DataExtractor, crop, augmentation
from evaluate import evaluate
from logger.logger import set_logger, info
from preprocess import downscale_image
from preprocess import load_dataset
from modules.SRGAN.train import train as srgan_train
from modules.SRGAN.test import test as srgan_test
from modules.SRGAN.test import predict as srgan_predict
from modules.TRAD.test import test_bicubic
from modules.TRAD.test import test_fd
from datasets import load_from_disk, load_dataset


set_logger('./logger/log.txt')
db_path = './dataset'
downscale_factor = 4
db_train_lr_path = f'./dataset/DIV2K_train_LRx{downscale_factor}'
db_valid_lr_path = f'./dataset/DIV2K_valid_LRx{downscale_factor}'
db_train_hr_path = './dataset/DIV2K_train_HR'
db_valid_hr_path = './dataset/DIV2K_valid_HR'


def downscale_images(lr_path, hr_path, downscale_factor):
    """
    Downscale HR images by a given factor using bicubic interpolation
    and saved in the given LR path.

    :param lr_path: Path of the LR images directory to be saved within.
    :param hr_path: Path of the HR images directory.
    :param downscale_factor: Factor by which to downscale the image.
    """
    N = len(os.listdir(hr_path))
    info(f'Loading {N} HR images from {hr_path}')
    hr_imgs = load_dataset(hr_path)
    info(f'Downscale of x{downscale_factor} has been started')
    for hr_img in hr_imgs:
        lr_img = downscale_image(img=hr_img['img'], factor=downscale_factor)
        lr_img.save(os.path.join(lr_path, os.path.basename(hr_img['path'])))
    info(f'Downscale of x{downscale_factor} has been completed\n'
         f'{N} images have been downscaled and saved in {lr_path}')


def args_handler(args):
    if args.model == 'srgan':
        args.db_valid_sr_path = f'./modules/SRGAN/results/{args.model_type.upper()}'
        pre_trained_dir = r'./modules/SRGAN/pre_trained'
        args.generator_path = os.path.join(pre_trained_dir,'SRGAN.pt') \
            if args.model_type == 'srgan' else os.path.join(pre_trained_dir, 'SRResNet.pt')

    if args.model == 'trad':
        args.db_valid_sr_path = f'./modules/TRAD/results/{args.model_type.upper()}'
    
    if args.model == 'srooe':
        args.db_valid_sr_path = './modules/SROOE/results/SROOE_t100/DIV2K_valid_LRx4'

    return args


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='Examine several SR modules and responsible on the pre-process.'
    )
    parser.add_argument("--model", type=str, default='srooe', help='srgan/srooe/trad')
    parser.add_argument("--model_type", type=str, default='srooe', help='|srgan: srgan/srresnet| trad: bicubic/fd|')
    parser.add_argument("--mode", type=str, default='evaluate', help='train/test/evaluate')
    parser.add_argument("--scale", type=int, default=4, help='Scale for each patch in an image')
    parser.add_argument("--patch_size", type=int, default=24, help='Number of patches for one image')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in DataLoader')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of the batch through train and test')
    parser.add_argument('--downscale', type=bool, default=False,
                        help='Create a LR version of train and valid dataset')
    parser.add_argument('--downscale_factor', type=int, default='4', help='Specify the downscale factor')
    parser.add_argument('--cuda', type=bool, default=True, help='Choose True or False for device cuda:0/cpu')
    parser.add_argument('--db_train_lr_path', type=str, default=db_train_lr_path)
    parser.add_argument('--db_valid_lr_path', type=str, default=db_valid_lr_path)
    parser.add_argument('--db_train_hr_path', type=str, default=db_train_hr_path)
    parser.add_argument('--db_valid_hr_path', type=str, default=db_valid_hr_path)
    parser.add_argument('--db_valid_sr_path', type=str, default=None)

    # SRGAN
    parser.add_argument("--fine_tuning", type=bool, default=True)
    parser.add_argument("--res_num", type=int, default=16)
    parser.add_argument("--L2_coeff", type=float, default=1.0)
    parser.add_argument("--adv_coeff", type=float, default=1e-3)
    parser.add_argument("--tv_loss_coeff", type=float, default=0.0)
    parser.add_argument("--pre_train_epoch", type=int, default=0)
    parser.add_argument("--fine_train_epoch", type=int, default=4000)
    parser.add_argument("--feat_layer", type=str, default='relu5_4')
    parser.add_argument("--vgg_rescale_coeff", type=float, default=0.006)
    parser.add_argument("--generator_path", type=str, default='./modules/SRGAN/pre_trained/SRGAN.pt')

    args, unknown = parser.parse_known_args()
    args = args_handler(args)

    if args.downscale:
        downscale_images(lr_path=db_train_lr_path, hr_path=db_train_hr_path, downscale_factor=downscale_factor)
        downscale_images(lr_path=db_valid_lr_path, hr_path=db_valid_hr_path, downscale_factor=downscale_factor)

    if args.mode == 'train':
        if args.model == 'srgan':
            info('SRGAN training is running')
            srgan_train(args)

    if args.mode == 'test':
        if args.model == 'trad':
            if args.model_type == 'bicubic':
                test_bicubic(args)

            if args.model_type == 'fd':
                test_fd(args)

        if args.model == 'srgan':
            info('SRGAN testing is running')
            srgan_test(args)
        
        if args.model == 'srooe':
            info('SROOE testing is running')
            subprocess.run(["python", r"modules/SROOE/codes/test.py", "-opt", r"modules/SROOE/codes/options/test/test.yml"])
    
    if args.mode == 'evaluate':
        evaluate(args)
