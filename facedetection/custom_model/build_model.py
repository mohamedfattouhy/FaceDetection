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
    """Create a custom neural network model for classification
    and regression from the VGG16 model (without including 3 fully-connected
    layers at the top of the network)"""

    input_layer = Input(shape=(120, 120, 3))  # Input image (120x120x3 px)

    # Feed the VGG16 model with the input image (without the top)
    vgg = VGG16(include_top=False)(input_layer)

    # Classification model to determine whether a face is present or not
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Régression for bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])

    return facetracker


def localization_loss(y_true, yhat):
    """Calculation of the loss associated
    with the regression (box coordinates)"""

    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]  # True height
    w_true = y_true[:, 2] - y_true[:, 0]  # True width

    h_pred = yhat[:, 3] - yhat[:, 1]  # predicted height
    w_pred = yhat[:, 2] - yhat[:, 0]  # predicted width

    # Compute the localization loss by adding the squared error
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred)
                               + tf.square(h_true-h_pred))

    return delta_coord + delta_size
