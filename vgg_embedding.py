import glob
import os

import h5py
import numpy as np
from joblib import Parallel, delayed
from PIL import Image as pil_image
from scipy.misc import imsave
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg

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


def load_from_directory(image_directory, n_samples=None, dtype=np.uint8, n_jobs=1):
    image_glob = os.path.join(image_directory, '*.jpg')
    image_files = glob.glob(image_glob)
    if n_samples is not None and n_samples < len(image_files):
        image_files = sample_images(image_files, n_samples)

    # perform this in parallel with joblib
    images = Parallel(n_jobs=n_jobs)(
                delayed(load_image)(img, target_size=(image_size, image_size), dtype=dtype)
                for img in image_files)

    return np.vstack(images)


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


def images_to_sprite(image_directory, n_samples=None):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    image_directory: str
        Path to the directory holding the images.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    data = load_from_directory(
        image_directory,
        n_samples=n_samples,
        dtype=np.float32,
        n_jobs=-1)

    # apply pixel-wise min/max scaling
    data = min_max_scale(data)

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


def vgg16_features(image_tensor, checkpoint_path, include_top=False, name=None):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        logits, end_points = vgg.vgg_16(image_tensor, is_training=False)

    init_vgg = slim.assign_from_checkpoint_fn(
        checkpoint_path,
        slim.get_model_variables('vgg_16'))

    output = logits if include_top else end_points['vgg_16/pool5']

    return init_vgg, output


def process_jpeg_image(image_file, name=None, target_size=244):
    """Read images from a file and resize to the appropriate input."""
    with tf.name_scope(name, 'process_image', [image_file]):
        image = tf.image.decode_jpeg(image_file, channels=3)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, [target_size, target_size])
        image = image  / 255.
        return tf.squeeze(image)


def vgg_embedding(image_directory, checkpoint_path, batch_size=32):
    with tf.Graph().as_default():
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(os.path.join(image_directory, '*.jpg')),
            num_epochs=1,
            shuffle=False)

        image_reader = tf.WholeFileReader()
        image_path, image_file = image_reader.read(filename_queue)
        image = process_jpeg_image(image_file,
                                   target_size=vgg.vgg_16.default_image_size)

        images, image_paths = tf.train.batch(
            [image, image_path],
            batch_size=batch_size,
            num_threads=1,  # vgg is run on gpu this can be higher
            allow_smaller_final_batch=True)

        init_vgg16, features = vgg16_features(images,
                                              checkpoint_path=checkpoint_path,
                                              include_top=False)

        init_op = tf.group(tf.local_variables_initializer(),
                           tf.global_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            init_vgg16(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():
                    yield sess.run([features, image_paths])
            except tf.errors.OutOfRangeError:
                pass
            finally:
                coord.request_stop()
                coord.join(threads)


def create_dset_from_chunk(h5file, chunk, name):
    maxshape = (None,) + chunk.shape[1:]

    if chunk.dtype == np.object:
        dt = h5py.special_dtype(vlen=unicode)
        chunk = chunk.astype(np.unicode)
    else:
        dt = chunk.dtype

    dset = h5file.create_dataset(name,
                                 shape=chunk.shape,
                                 maxshape=maxshape,
                                 chunks=chunk.shape,
                                 dtype=dt,
                                 compression='gzip',
                                 compression_opts=9)

    dset[:] = chunk

    return dset


def expand_dset_from_chunk(dset, chunk, row_count):
    dset.resize(row_count + chunk.shape[0], axis=0)
    dset[row_count:] = chunk


def create_from_generator(file_name, generator):
    with h5py.File(file_name, 'w') as h5file:
        feature_chunk, file_chunk = next(generator)
        row_count = feature_chunk.shape[0]

        feature_dset = create_dset_from_chunk(h5file, feature_chunk, 'vgg_features')
        file_dset = create_dset_from_chunk(h5file, file_chunk, 'file_paths')

        for chunk in generator:
            expand_dset_from_chunk(feature_dset, feature_chunk, row_count)
            expand_dset_from_chunk(file_dset, file_chunk, row_count)
            row_count += feature_chunk.shape[0]


#sprite = images_to_sprite('./images', n_samples=500)
#vgg_gen = vgg_embedding('./images', './vgg_16.ckpt', batch_size=32)
#create_from_generator('vgg_features.hdf5', vgg_gen)

#with tf.Session() as sess:
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#    try:
#        with coord.stop_on_exception():
#            while not coord.should_stop():
#                print sess.run(target)
#    finally:
#        coord.join(threads)
