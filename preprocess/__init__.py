import os
import glob
from PIL import Image


def load_dataset(dataset_path):
    """
    Load the images and return a generator to save memory.

    :param dataset_path: Path to the directory of the input images.
    """
    image_paths = glob.glob(os.path.join(dataset_path, '*.png'))
    for image_path in image_paths:
        yield {'path': image_path, 'img': Image.open(image_path)}


def downscale_image(img, factor):
    """
    Downscale the image by a given factor using bicubic interpolation.

    :param img: Image object of the input image.
    :param factor: Factor by which to downscale the image.
    """
    # Calculate the new size
    new_size = (img.width // factor, img.height // factor)

    # Resize the image using bicubic interpolation
    downscaled_img = img.resize(new_size, Image.BICUBIC)

    return downscaled_img
