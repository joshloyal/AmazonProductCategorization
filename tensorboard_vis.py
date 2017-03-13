import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import image_utils
from resnet import ResNetVectorizer

log_dir = '/tmp/dumm'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

df = pd.read_csv('amazon_products_train.csv')
n_samples = 500
image_list = df.image_file.values[:n_samples]
target = df.product_category.values[:n_samples]

# make sprite image
images = image_utils.load_images(image_list,
                                 image_dir='images',
                                 target_size=[244, 244],
                                 dtype=np.float32)
sprite_image = image_utils.images_to_sprite(images, as_image=True)
sprite_image.save(os.path.join(log_dir, 'sprite.png'))

# make metadata file
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as metadata_file:
    metadata_file.write('{}\t{}\n'.format('index', 'class_name'))
    for index, target_value in enumerate(target.tolist()):
        metadata_file.write('{:d}\t{}\n'.format(index, target_value))

vec = ResNetVectorizer(image_dir='images', use_cache=True, cache_dir='resnet50')
features = vec.fit_transform(image_list)


embedding_var = tf.Variable(features, name='embedding')
summary_writer = tf.summary.FileWriter(log_dir)
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# add target metadata
embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')

# add sprites
embedding.sprite.image_path = os.path.join(log_dir, 'sprite.png')
image_size = images.shape[1]
embedding.sprite.single_image_dim.extend([image_size, image_size])

projector.visualize_embeddings(summary_writer, config)
saver = tf.train.Saver([embedding_var])

with tf.Session() as sess:
    sess.run(embedding_var.initializer)
    saver.save(sess, os.path.join(log_dir, 'model2.ckpt'), 1)
