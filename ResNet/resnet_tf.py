
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
from tensorflow.keras import layers


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
        x = self.max_pool(x)

        x = self.idla(x)
        x = self.idlb(x)

        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def summary_model(self, input_shape=(224, 224, 3)):
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        tf.keras.Model(inputs=inputs, outputs=outputs).summary()

if __name__ == "__main__":

    img_size = 224
    num_classes = 10
    x = tf.random.uniform(shape=[1, img_size, img_size, 3])
    model = ResNet(num_classes)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}\n")
    model.build((1, img_size, img_size, 3))
    print(model.summary_model(input_shape=(img_size, img_size, 3)))
