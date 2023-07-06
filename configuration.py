# MANAGE ENVIRONMENT
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from facedetection.collect_data.collect_data import collect_data
from facedetection.collect_data.create_folder import create_folder
from facedetection.collect_data.augment_data import augment_data
from facedetection.collect_data.rectangle_detection import display_rectangle_detection
from facedetection.preprocess.preprocess_data import (decode_and_preprocess,
                                                      label_dataset,
                                                      load_preprocess_data)
from facedetection.custom_model.build_model import build_model, localization_loss
from facedetection.custom_model.custom_model_class import FaceDetection
import facedetection.path as path


# Create folders and subfloders to store the data
create_folder(path.DATA_DIR_PATH, ['train', 'test', 'validation'])
create_folder(path.TRAIN_DIR_PATH, ['image', 'labelled_images'])
create_folder(path.TEST_DIR_PATH, ['image', 'labelled_images'])
create_folder(path.VALIDATION_DIR_PATH, ['image', 'labelled_images'])

# Create folders and subfloders to store the augmented data
create_folder(path.AUGMENTED_DATA_DIR_PATH, ['train', 'test', 'validation'])
create_folder(path.AUGMENTED_TRAIN_DIR_PATH, ['image', 'labelled_images'])
create_folder(path.AUGMENTED_TEST_DIR_PATH, ['image', 'labelled_images'])
create_folder(path.AUGMENTED_VALIDATION_DIR_PATH, ['image', 'labelled_images'])

NUMBER_IMAGES = 10
NUMBER_AUG_DATA = 30

# Collect data and put them into data folder
collect_data(path.TRAIN_IMAGES_PATH, NUMBER_IMAGES)
collect_data(path.TEST_IMAGES_PATH, NUMBER_IMAGES)
collect_data(path.VALIDATION_IMAGES_PATH, NUMBER_IMAGES)

# Augment data images and put them into augmented_data folder
directories = ['train', 'test', 'validation']
augment_data(NUMBER_AUG_DATA, directories)

files = os.listdir(path.TRAIN_IMAGES_PATH)
img_sample_path = os.path.join(path.TRAIN_IMAGES_PATH, files[0])

json_label_files = os.listdir(os.path.join(path.TRAIN_DIR_PATH,
                                           'labelled_images'))
json_path_label = os.path.join(path.TRAIN_DIR_PATH,
                               'labelled_images', json_label_files[0])

# Display an image with the box
display_rectangle_detection(img_sample_path, json_path_label)


# Load augmented images to Tensorflow dataset
train_images_path = os.path.join(path.AUGMENTED_TRAIN_DIR_PATH,
                                 'images', '*.jpg')
test_images_path = os.path.join(path.AUGMENTED_TEST_DIR_PATH,
                                'images', '*.jpg')
validation_images_path = os.path.join(path.AUGMENTED_VALIDATION_DIR_PATH,
                                      'images', '*.jpg')

train_images = decode_and_preprocess(train_images_path)
test_images = decode_and_preprocess(test_images_path)
validation_images = decode_and_preprocess(validation_images_path)

train_images = decode_and_preprocess(train_images_path)
test_images = decode_and_preprocess(train_images_path)
validation_images = decode_and_preprocess(train_images_path)


# Load labels to Tensorflow dataset
train_labels_path = os.path.join(path.AUGMENTED_TRAIN_DIR_PATH,
                                 'labelled_images', '*.json')
test_labels_path = os.path.join(path.AUGMENTED_TEST_DIR_PATH,
                                'labelled_images', '*.json')
validation_labels_path = os.path.join(path.AUGMENTED_VALIDATION_DIR_PATH,
                                      'labelled_images', '*.json')

train_labels = label_dataset(train_labels_path)
test_labels = label_dataset(test_labels_path)
validation_labels = label_dataset(validation_labels_path)

# Load pre-processed data
train = load_preprocess_data(train_images, train_labels, 100, 20, 10)
test = load_preprocess_data(test_images, test_labels, 25, 10, 5)
validation = load_preprocess_data(validation_images,
                                  validation_labels, 25, 10, 5)

# View images and annotations
data_samples = train.as_numpy_iterator()
res = data_samples.next()

fig, ax = plt.subplots(ncols=2, figsize=(20, 20))

for idx in range(0, 2):

    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(img=sample_image,
                  pt1=tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  pt2=tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  color=(255, 0, 0),
                  thickness=1)

    ax[idx].imshow(sample_image)

plt.show()

# Define losses and optimizers
batches_per_epoch = len(train)
lr_decay = (1./0.75 - 1)/batches_per_epoch
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)
classloss = tf.keras.losses.BinaryCrossentropy()  # Classification
regressloss = localization_loss  # Regression (coords)

# Create a model
facetracker = build_model()
model = FaceDetection(facetracker)

# Compile the model
model.compile(opt, classloss, regressloss)

# Fit the model
hist = model.fit(train, epochs=10, validation_data=validation)

# Plot Performances
fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

ax[0].plot(hist.history['total_loss'], color='teal', label='total loss')
ax[0].plot(hist.history['val_total_loss'],
           color='orange', label='validation loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['classification_loss'], color='teal',
           label='classification loss')
ax[1].plot(hist.history['val_classification_loss'],
           color='orange', label='validation class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regression_loss'], color='teal',
           label='regression loss')
ax[2].plot(hist.history['val_regression_loss'],
           color='orange', label='validation regression loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()

# Make Predictions on Test Set
test_data = test.as_numpy_iterator()
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))

for idx in range(0, 4):

    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.5:

        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                      (255, 0, 0),
                       2)

    ax[idx].imshow(sample_image)
