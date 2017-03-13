from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import glob
import os

import numpy as np
from joblib import Parallel, delayed
from PIL import Image as pil_image


image_size = 128


def sample_images(seq, n_samples, seed=123):
    random_state = np.random.RandomState(seed)
    return random_state.choice(seq, size=n_samples, replace=False)


def load_image(image_path, target_size=None, dtype=np.uint8):
    """Loads an image into PIL format."""
    img = pil_image.open(image_path).convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return np.expand_dims(np.asarray(img, dtype), 0)


def load_images(image_files, n_samples=None, dtype=np.uint8, n_jobs=1):
    if n_samples is not None and n_samples < len(image_files):
        image_files = sample_images(image_files, n_samples)

    # perform this in parallel with joblib
    images = Parallel(n_jobs=n_jobs)(
                delayed(load_image)(img, target_size=(image_size, image_size), dtype=dtype)
                for img in image_files)

    return np.vstack(images)


def load_from_directory(image_directory, n_samples=None, dtype=np.uint8, n_jobs=1):
    image_glob = os.path.join(image_directory, '*.jpg')
    image_files = glob.glob(image_glob)
    return load_images(image_files, n_samples=n_samples, dtype=dtype, n_jobs=n_jobs)


def min_max_scale(data):
    """Apply a row-wise min-max scaling to an nd-array."""
    # save the original shape (since will need to reshape later)
    original_shape = data.shape

    # flatten the array
    data = data.reshape((data.shape[0], -1))

    # rowwise apply (x - min_x) / max_x
    pixel_min = np.min(data, axis=1)[:, np.newaxis]
    pixel_max = np.max(data, axis=1)[:, np.newaxis]
    data -= pixel_min
    data /= pixel_max

    return data.reshape(original_shape)


def zero_pad(data, target_size):
    padding_size = target_size ** 2 - data.shape[0]
    padding = ((0, padding_size),) + ((0, 0),) * (data.ndim - 1)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)

    return data


def images_to_sprite(images):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    images : array-like of shape [n_samples, width, height, channels]
        A four dimensional array of images.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    # apply pixel-wise min/max scaling
    data = min_max_scale(images)

    # sprite image should be sqrt(n_samples) x sqrt(n_samples)
    # this means we need to pad the first dimension (the samples)
    # to make this an even square.
    target_size = int(np.ceil(np.sqrt(data.shape[0])))
    data = zero_pad(data, target_size)

    # Tile the individual thumbnails into an image
    data = data.reshape((target_size, target_size) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((target_size * data.shape[1],
                         target_size * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)

    return data


def directory_to_sprites(image_directory, n_samples=None, n_jobs=1):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    image_directory : str
        Path to the directory holding the images.

    n_samples : int (default=None)
        The number of random sample images to use. If None, then
        all images are loaded. This can be memory expensive.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    images = load_from_directory(
        image_directory,
        n_samples=n_samples,
        dtype=np.float32,
        n_jobs=n_jobs)

    return images_to_sprite(images)
