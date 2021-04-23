
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss
from keras import backend as K

# Wrapper loss
def constrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):

        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))

        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Object loss
class my_contrastive_loss_with_margin(Loss):
    margin = 1

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))

        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
