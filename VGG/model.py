
import tensorflow as tf 
import tf.keras.layers as layers
import tf.keras.Model as Model

class Block(Model):
    def __init__(self, filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.repetitions = repetitions

        for i in range(self.repetitions):
            vars(self)[f"conv2D_{i}"] = layers.Conv2D(self.filters, self.kernel_size, activation="relu", padding="same")

        self.max_pool = layers.MaxPoolong2D(pool_size, strides)

    def call(self,  inputs):
        conv2D_0 = vars(self)["conv2D_0"]
        x = conv2D_0(inputs)

        for i in range(1, self.repetitions):
            conv2D_i = vars(self)[f"conv2D_{i}"]
            x = conv2D_i(x)

        max_pool = self.max_pool(x)
        return max_pool


class VGG(Model):
    def __init__(self,  num_classes):
        super(VGG, self).__init__()
        self.block_a = Block(64, 3, 2)
        self.block_b = Block(128, 3, 2)
        self.block_c = Block(256, 3, 3)
        self.block_d = Block(512, 3, 3)
        self.block_e = Block(512, 3, 3)

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(256, activation="relu")
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x