import pandas as pd
import tensorflow as tf
from tensorflow.contrib import learn


def process_jpeg_image(image_file, name=None, target_size=244):
    """Read images from a file and resize to the appropriate input."""
    with tf.name_scope(name, 'process_image', [image_file]):
        image = tf.image.decode_jpeg(image_file, channels=3)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, [target_size, target_size])
        image = (image  / 255.) - 0.5
        return tf.squeeze(image)


#X = pd.read_csv('amazon_products.csv')
#y = X.pop('product_category')
#
#input_fn = learn.pandas_input_fn(X, y,
#                                 shuffle=False,
#                                 batch_size=32,
#                                 num_threads=1,
#                                 target_column='product_category')

def input_function(csv_location, image_dir='./images/', batch_size=32):
    csv_queue = tf.train.string_input_producer([csv_location])
    csv_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = csv_reader.read(csv_queue)
    title, image_file, target = tf.decode_csv(csv_content, record_defaults=[[""], [""], [""]])

    image_content = tf.read_file(image_dir + image_file)
    image = process_jpeg_image(image_content)

    # make batches
    title_batch, image_batch, target_batch = tf.train.batch([title, image, target], batch_size=batch_size)

    return {'title': title_batch, 'image': image_batch}, target_batch


with tf.Graph().as_default():
    features, target = input_function('amazon_products.csv')
    #features, target = input_fn()

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        while not sv.should_stop():
            print sess.run(features['images'])



