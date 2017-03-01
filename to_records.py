import numpy as np
import pandas as pd
import tensorflow as tf


from PIL import Image

"""
Cool way to load a dataframe
from tensorflow.contrib import learn
df = learn.TensorFlowDataFrame(['amazon_products.csv'], [[''], [''], ['']])
"""

def to_bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_binary_image(filename):
    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return image.tobytes(), shape.tobytes()  # convert image to raw data bytes in the array


def to_example(title, label, shape, binary_image):
    return tf.train.Example(features=tf.train.Features(feature={
        'title': to_bytes_feature(title),
        'label': to_bytes_feature(label),
        'shape': to_bytes_feature(shape),
        'image': to_bytes_feature(binary_image),
    }))



def csv_to_tfrecord(filename, tfrecord_file):
    dataframe = pd.read_csv(filename).dropna()
    writer = tf.python_io.TFRecordWriter(tfrecord_file)

    for _, row in dataframe.iterrows():
        # text columns
        title, label = row['title'], row['product_category']

        # load images from the file path
        image, shape = get_binary_image(row['image_location'])

        example = to_example(title, label, shape, image)

        writer.write(example.SerializeToString())

    writer.close()


def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                            features={
                                'label': tf.FixedLenFeature([], tf.string),
                                'shape': tf.FixedLenFeature([], tf.string),
                                'image': tf.FixedLenFeature([], tf.string),
                                'title': tf.FixedLenFeature([], tf.string),
                            }, name='features')

    # image was saved as uint8, so we have to decode to uint8
    image = tf.decode_raw(tfrecord_features['image'], tf.unit7)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)

    # The image tensor is flattend out, so we have to reconstruct the shape
    image = tf.reshape(image, shape)
    label = tf.cast(tfrecord_features['label'], tf.string)

    return label, shape, image

# csv_to_tfrecord('amazon_products.csv', 'amazon_products.tfrecord')
