# MANAGE ENVIRONMENT
import tensorflow as tf
from keras.models import Model


# Create custom model class derived from Model
class FaceDetection(Model):

    def __init__(self, eyetracker):
        super().__init__()
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss):
        super().compile()
        self.class_loss = classloss
        self.localization_loss = localizationloss
        self.opt = opt

    def train_step(self, batch):

        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_class_loss = self.class_loss(y[0], classes)
            batch_localization_loss = self.localization_loss(tf.cast(y[1], tf.float32),
                                                             coords)

            total_loss = batch_localization_loss + 0.5*batch_class_loss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_class_loss,
                "regress_loss": batch_localization_loss}

    def test_step(self, batch):

        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_class_loss = self.class_loss(y[0], classes)
        batch_localization_loss = self.localization_loss(tf.cast(y[1], tf.float32),
                                                         coords)
        total_loss = batch_localization_loss + 0.5*batch_class_loss

        return {"total_loss": total_loss, "class_loss": batch_class_loss,
                "regress_loss": batch_localization_loss}
