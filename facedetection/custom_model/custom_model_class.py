# MANAGE ENVIRONMENT
import tensorflow as tf
from keras.models import Model


# Create custom model class which inherits from the Model (keras class).
# This means that the "FaceDetection" class can use the attributes
# and methods defined in the Model class, as well as defining
# our own attributes and methods specific to face detection.
class FaceDetection(Model):
    """Create custom model class which inherits from the Model class (keras class)
    for a face dectection model"""

    def __init__(self, facetracker) -> None:
        super().__init__()  # call the constructor method of the Model class of keras
        self.model = facetracker

    def compile(self, opt, classloss, localizationloss) -> None:
        super().compile()  # call the compile() method of the Model class of keras
        self.class_loss = classloss
        self.localization_loss = localizationloss
        self.opt = opt

    def train_step(self, batch) -> dict:
        """Train the face detection model on a bacth by calculating
        the total loss gradient and updating the model weights"""

        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_class_loss = self.class_loss(y[0], classes)
            batch_localization_loss = self.localization_loss(tf.cast(y[1], tf.float32),
                                                             coords)

            total_loss = batch_localization_loss + 0.5*batch_class_loss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "classification_loss": batch_class_loss,
                "regression_loss": batch_localization_loss}

    def test_step(self, batch) -> dict:
        """Test the trained face detection model on a bacth"""

        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_class_loss = self.class_loss(y[0], classes)
        batch_localization_loss = self.localization_loss(tf.cast(y[1], tf.float32),
                                                         coords)
        total_loss = batch_localization_loss + 0.5*batch_class_loss

        return {"total_loss": total_loss, "classification_loss": batch_class_loss,
                "regression_loss": batch_localization_loss}
