
import tensorflow as tf 
from tf.keras.layers import Layer 
from tf.keras.model import Model
from tf.keras import layers


class IdentityBlock(Model):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()

        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same")
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.act  = layers.Activation("relu")
        self.add = layers.Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)
        return x


class ResNet(Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = layers.Conv2D(64, 7, padding="same")
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("relu")
        self.max_pool = layers.MaxPool2D((3,3))

        self.idla = IdentityBlock(64, 3)
        self.idlb = IdentityBlock(64, 3)

        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_poool(x)

        x = self.idla(x)
        x = self.idlb(x)

        x = self.global_pool(x)
        return x