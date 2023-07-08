# MANAGE ENVIRONMENT
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GlobalMaxPooling2D
from keras.applications import VGG16

# Download VGG16 model
vgg = VGG16(include_top=False)
# print()
# print(vgg.summary())


def build_model() -> Model:
    """Create a custom neural network model from the
    VGG16 model, for classification and regression."""

    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])

    return facetracker


def localization_loss(y_true, yhat):
    """Calculation of the loss associated
    with the regression (box coordinates)"""

    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred)
                               + tf.square(h_true-h_pred))

    return delta_coord + delta_size
