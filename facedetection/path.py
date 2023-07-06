# MANAGE ENVIRONMENT
import os

# Storage of the various paths required

DATA_DIR_PATH = os.path.join('facedetection', 'data')
TRAIN_DIR_PATH = os.path.join('facedetection', 'data', 'train')
TEST_DIR_PATH = os.path.join('facedetection', 'data', 'test')
VALIDATION_DIR_PATH = os.path.join('facedetection', 'data', 'validation')


AUGMENTED_DATA_DIR_PATH = os.path.join('facedetection', 'augmented_data')
AUGMENTED_TRAIN_DIR_PATH = os.path.join('facedetection',
                                        'augmented_data', 'train')
AUGMENTED_TEST_DIR_PATH = os.path.join('facedetection',
                                       'augmented_data', 'test')
AUGMENTED_VALIDATION_DIR_PATH = os.path.join('facedetection',
                                             'augmented_data', 'validation')

TRAIN_IMAGES_PATH = os.path.join('facedetection', 'data', 'train', 'images')
TEST_IMAGES_PATH = os.path.join('facedetection', 'data', 'test', 'images')
VALIDATION_IMAGES_PATH = os.path.join('facedetection', 'data',
                                      'validation', 'images')
