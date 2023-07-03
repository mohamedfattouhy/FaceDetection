# MANAGE ENVIRONMENT
import json
import tensorflow as tf


# Decode images
def decode_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


# Read data and pre-process it
def decode_and_preprocess(data_path):

    dataset_images = tf.data.Dataset.list_files(data_path, shuffle=False)
    dataset_images = dataset_images.map(decode_image)
    dataset_images = dataset_images.map(lambda x: tf.image.resize(x, (120, 120)))
    dataset_images = dataset_images.map(lambda x: x/255)

    return dataset_images


# Build Label Loading Function
def load_labels(label_path):

    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']


# Load Labels to Tensorflow Dataset
def label_dataset(labelled_images_path):

    dataset_labels = tf.data.Dataset.list_files(labelled_images_path,
                                                shuffle=False)
    dataset_labels = dataset_labels.map(lambda x: tf.py_function(load_labels, [x],
                                        [tf.uint8, tf.float16]))

    return dataset_labels


# Create pre-processed data
def load_preprocess_data(image_dataset, labels_dataset,
                         n_shuffle, n_batch, n_prefetch):

    preprocessed_dataset = tf.data.Dataset.zip((image_dataset, labels_dataset))
    preprocessed_dataset = preprocessed_dataset.shuffle(n_shuffle)
    preprocessed_dataset = preprocessed_dataset.batch(n_batch)
    preprocessed_dataset = preprocessed_dataset.prefetch(n_prefetch)

    return preprocessed_dataset
