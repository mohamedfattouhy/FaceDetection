# MANAGE ENVIRONMENT
import json
import tensorflow as tf


def decode_image(x):
    """Read a JPEG-encoded image and decode it
    to a uint8 tensor"""
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img


def decode_and_preprocess(data_path: str):
    """Read data and pre-process thm by resizing
    and normalizing them"""

    dataset_images = tf.data.Dataset.list_files(data_path, shuffle=False)
    dataset_images = dataset_images.map(decode_image)
    dataset_images = dataset_images.map(lambda x: tf.image.resize(x, (120, 120)))
    dataset_images = dataset_images.map(lambda x: x/255)

    return dataset_images


def load_labels(label_path: str) -> tuple:
    """Get the class and the box coordinates
    from the labelled images (from json format)"""

    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']


def label_dataset(labelled_images_path: str):
    """Load labels to tensorflow dataset"""

    dataset_labels = tf.data.Dataset.list_files(labelled_images_path,
                                                shuffle=False)
    dataset_labels = dataset_labels.map(lambda x: tf.py_function(load_labels, [x],
                                        [tf.uint8, tf.float16]))

    return dataset_labels


def load_preprocessed_data(image_dataset, labels_dataset,
                           n_shuffle: int, n_batch: int, n_prefetch: int):
    """Load images and labels and associate them in a single dataset.
    Then configure data by defining batch, shuffle and prefetch
    to train or test a model"""

    preprocessed_dataset = tf.data.Dataset.zip((image_dataset, labels_dataset))
    preprocessed_dataset = preprocessed_dataset.shuffle(n_shuffle)
    preprocessed_dataset = preprocessed_dataset.batch(n_batch)
    preprocessed_dataset = preprocessed_dataset.prefetch(n_prefetch)

    return preprocessed_dataset
