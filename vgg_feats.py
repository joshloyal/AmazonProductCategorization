from __future__ import division

import csv
import cytoolz
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.estimator.inputs.queues import feeding_functions


def process_jpeg_image(image_file, name=None, target_size=244):
    """Read images from a file and resize to the appropriate input."""
    with tf.name_scope(name, 'process_image', [image_file]):
        image = tf.image.decode_jpeg(image_file, channels=3)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, [target_size, target_size])
        image = (image  / 255.) - 0.5
        return tf.squeeze(image)


df = pd.read_csv('amazon_products.csv')


def pandas_input_fn(dataframe,
                    batch_size=128,
                    num_epochs=1,
                    shuffle=True,
                    queue_capacity=1000,
                    num_threads=1,
                    target_column=None,
                    image_columns=None):
    x = dataframe.copy()

    if target_column is not None and target_column not in dataframe:
        raise ValueError('Target column %s specified but not in '
                         'DataFrame.' % (target_column))

    if queue_capacity is None:
        if shuffle:
            queue_capacity = 4 * len(x)
        else:
            queue_capacity = len(x)
    min_after_dequeue = max(queue_capacity / 4, 1)

    def input_fn():
        """Pandas input function."""
        queue = feeding_functions._enqueue_data(
            x,
            queue_capacity,
            shuffle=shuffle,
            min_after_dequeue=min_after_dequeue,
            num_threads=num_threads,
            enqueue_size=batch_size,
           num_epochs=num_epochs)
        if num_epochs is None:
            features = queue.dequeue_many(batch_size)
        else:
            features = queue.dequeue_up_to(batch_size)
        assert len(features) == len(x.columns) + 1, ('Features should have one '
                                                     'extra element for the index.')
        features = features[1:]
        features = dict(zip(list(x.columns), features))
        if target_column is not None:
            target = features.pop(target_column)
            return features, target
        return features
    return input_fn


def get_default(dtype):
    if np.issubdtype(dtype, np.number):
        return [0.0]
    return ['']


def get_record_defaults(csv_path):
    """Uses pandas to read a chunk of a csv to determine data types."""
    dtype_list = pd.read_csv(csv_path, nrows=100).dtypes.tolist()
    return [get_default(dtype) for dtype in dtype_list]


def read_columns_from_header(csv_path):
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        return reader.next()


def make_image_queue(queue,
                     image_dir):
    image_content = tf.read_file(image_dir + queue)
    image = process_jpeg_image(image_content)
    return image


def extract_image_queues(queues,
                         column_names,
                         image_columns):

    image_queue, feature_queue = [], []
    for name, queue in zip(column_names, queues):
        if name in image_columns:
            image_queue.append((name,
                                make_image_queue(queue, './images/')))
        else:
            feature_queue.append((name, queue))

    return image_queue, feature_queue


def input_function(csv_path,
                   has_header=True,
                   column_names=None,
                   image_dir='./images/',
                   record_defaults=None,
                   batch_size=128,
                   num_epochs=1,
                   shuffle=True,
                   queue_capacity=1000,
                   num_threads=1,
                   target_column=None,
                   image_columns=None):

    if record_defaults is None:
        record_defaults = get_record_defaults(csv_path)

    if has_header and column_names is None:
        column_names = read_columns_from_header(csv_path)

    # construct queues reading features from CSV
    csv_queue = tf.train.string_input_producer([csv_path])
    skip_header_lines = 1 if has_header else 0
    csv_reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
    _, csv_content = csv_reader.read(csv_queue)
    features = tf.decode_csv(csv_content, record_defaults=record_defaults)

    # split queues necessary for reading image files
    if image_columns:
        images, features = extract_image_queues(features,
                                                column_names,
                                                image_columns)

        image_columns, images = zip(*images)
        feature_columns, features = zip(*features)

        # make batches
        features_batch = tf.train.batch(
            features + images,
            batch_size=batch_size,
            #num_epochs=num_epochs,
            num_threads=num_threads,
            capacity=queue_capacity)
        features = dict(zip(feature_columns + image_columns, features_batch))
    else:
        features_batch = tf.train.batch(
            features,
            batch_size=batch_size,
            #num_epochs=num_epochs,
            num_threads=num_threads,
            capacity=queue_capacity)
        features = dict(zip(column_names, features_batch))

    if target_column:
        target = features.pop(target_column)
        return features, target
    return features


with tf.Graph().as_default():
    features, target = input_function('amazon_products.csv',
                                      target_column='product_category',
                                      #image_columns=['image_file'],
                                      num_epochs=1)

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        while not sv.should_stop():
            print(sess.run(target))
