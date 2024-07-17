import os
import argparse
from logger.logger import set_logger, info
from preprocess import downscale_image
from preprocess import load_dataset


set_logger('./logger/log.txt')
db_path = './dataset'
db_train_hr_path = './dataset/DIV2K_train_HR'
db_valid_hr_path = './dataset/DIV2K_valid_HR'
downscale_factor = 4
db_train_lr_path = f'./dataset/DIV2K_train_LRx{downscale_factor}'
db_valid_lr_path = f'./dataset/DIV2K_valid_LRx{downscale_factor}'


def downscale_images(hr_path, lr_path, downscale_factor):
    """
    Downscale HR images by a given factor using bicubic interpolation
    and saved in the given LR path.

    :param hr_path: Path of the HR images directory.
    :param lr_path: Path of the LR images directory to be saved within.
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


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='Examine several SR models and responsible on the pre-process.'
    )
    parser.add_argument('--downscale', type=bool, default='False',
                        help='Create a LR version of train and valid dataset')
    parser.add_argument('--downscale_factor', type=int, default='4', help='Specify the downscale factor')
    parser.add_argument('--device', type=str, default='cuda:0', help='Choose device cuda:0/cpu')

    args, unknown = parser.parse_known_args()

    if args.downscale:
        downscale_images(hr_path=db_train_hr_path, lr_path=db_train_lr_path, downscale_factor=downscale_factor)
        downscale_images(hr_path=db_valid_hr_path, lr_path=db_valid_lr_path, downscale_factor=downscale_factor)
