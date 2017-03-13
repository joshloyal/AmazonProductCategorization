import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from keras.applications import resnet50
from keras.preprocessing import image
import functools
import joblib
import cPickle as pickle
import chest
from generators import ImageListIterator


def get_cache(cache_dir):
    joblib_dump = functools.partial(joblib.dump,
                                    compress=True,
                                    protocol=pickle.HIGHEST_PROTOCOL)
    return chest.Chest(path=cache_dir,
                       dump=joblib_dump,
                       load=joblib.load)


def extract_filename(file_path):
    """Extracts a file's name without the extension.

    Parameters
    ----------
    file_path: str
        The path to the image file.

    Returns
    -------
    filename: str
        The name of the filename with path and extension information
        stripped, e.g, 'path/to/file.txt' returns 'file'.
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def split_cache_streams(image_list, cache):
    cache_indices, cache_files = [], []
    image_indices, image_files = [], []
    for index, image_file in enumerate(image_list):
        image_filename = extract_filename(image_file)
        if image_filename in cache:
            cache_indices.append(index)
            cache_files.append(image_filename)
        else:
            image_indices.append(index)
            image_files.append(image_file)

    return (cache_indices, cache_files), (image_indices, image_files)


def fill_from_cache(output_array, cache_files, cache_indices, cache):
    for index, cache_file in zip(cache_indices, cache_files):
        output_array[index, :] = cache[cache_file]


def write_to_cache(features, image_files, cache):
    for index, image_file in enumerate(image_files):
        image_filename = extract_filename(image_file)
        feature = features[index, :]
        cache[image_filename] = feature


df = pd.read_csv('amazon_products_train.csv')
image_list = df.image_file.apply(lambda x: './images/' + x).values


def extract_resnet50_features(image_list):
    image_size = 244
    n_channels = 3
    model = resnet50.ResNet50(include_top=False,
                              weights='imagenet',
                              input_shape=(image_size, image_size, n_channels))

    datagen = ImageListIterator(image_list, y=None,
                                image_data_generator=image.ImageDataGenerator(rescale=1./255.),
                                target_size=(image_size, image_size),
                                batch_size=32,
                                shuffle=False)

    return model.predict_generator(datagen, len(image_list))


class ResNetVectorizer(BaseEstimator, TransformerMixin):
    """Simple scikit-learn style transform that passes images through a
    resnet model. Has an option to cache the images for subsequent calls
    to the transformer."""
    def __init__(self, use_cache=False, cache_dir=None):
        self.cache_dir = cache_dir
        self.use_cache = use_cache

    def clear_cache(self):
        pass

    def fit(self, image_files):
        return self

    def transform(self, image_files):
        n_samples = len(image_files)

        if self.use_cache:
            cache = get_cache(self.cache_dir)
            (cache_indices, cache_files), (image_indices, image_files) = (
                split_cache_streams(image_files, cache))

        output = np.zeros((n_samples, 2048), dtype=np.float32)
        if image_files:
            features = np.squeeze(extract_resnet50_features(image_files))
            output[image_indices, :] = features

            if self.use_cache:
                write_to_cache(features, image_files, cache)

        if self.use_cache and cache_files:
            fill_from_cache(output, cache_files, cache_indices, cache)

        return output


vec = ResnetVectorizer(use_cache=True, cache_dir='resnet50')
features = vec.fit_transform(image_list)
