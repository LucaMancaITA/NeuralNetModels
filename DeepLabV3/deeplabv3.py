"""Module providing the DeeplabV3+ network architecture as a tf.keras.Model.
"""

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


BACKBONES = {
    'resnet50': {
        'model': tf.keras.applications.ResNet50,
        'feature_1': 'conv4_block6_2_relu',
        'feature_2': 'conv2_block3_2_relu'
    },
    'mobilenetv2': {
        'model': tf.keras.applications.MobileNetV2,
        'feature_1': 'out_relu',
        'feature_2': 'block_3_depthwise_relu'
    }
}


class Conv(tf.keras.layers.Layer):
    """Convolutional layer.

    Args:
        Layer (tf.keras.layers.Layer): TensorFlow layer.
    """

    def __init__(self, n_filters, kernel_size=3, non_linearity='lrelu', **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.non_linearity = non_linearity
        self.conv = tf.keras.layers.Conv2D(
            filters=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None,
        )
        if self.non_linearity == 'lrelu':
            self.act = tf.keras.layers.LeakyReLU(0.4)
        elif self.non_linearity == 'prelu':
            self.act = tf.keras.layersPReLU(shared_axes=[1, 2])
        else:
            self.act = tf.keras.layersActivation(self.non_linearity)

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.act(outputs)
        return outputs


class ConvBlock(tf.keras.layers.Layer):
    """Convolutional Block for DeepLabV3+
    Convolutional block consisting of Conv2D -> BatchNorm -> ReLU
    Args:
        n_filters:
            number of output filters
        kernel_size:
            kernel_size for convolution
        padding:
            padding for convolution
        kernel_initializer:
            kernel initializer for convolution
        use_bias:
            boolean, whether of not to use bias in convolution
        dilation_rate:
            dilation rate for convolution
        activation:
            activation to be used for convolution
    """
    # !pylint:disable=too-many-arguments
    def __init__(self, n_filters, kernel_size, padding, dilation_rate,
                 kernel_initializer, use_bias, conv_activation=None):
        super(ConvBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            n_filters, kernel_size=kernel_size, padding=padding,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias, dilation_rate=dilation_rate)

        self.batch_norm = tf.keras.layers.BatchNormalization()

        if conv_activation == "lrelu":
            self.act = tf.keras.layers.LeakyReLU(0.4)
        elif conv_activation == "prelu":
            self.act = tf.keras.layers.PReLU()
        else:
            self.act = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        tensor = self.conv(inputs)
        tensor = self.batch_norm(tensor)
        tensor = self.act(tensor)
        return tensor


class AtrousSpatialPyramidPooling(tf.keras.layers.Layer):
    """Atrous Spatial Pyramid Pooling layer for DeepLabV3+ architecture."""
    # !pylint:disable=too-many-instance-attributes
    def __init__(self, input_shape):
        super(AtrousSpatialPyramidPooling, self).__init__()

        # layer architecture components
        dummy_tensor = tf.random.normal(input_shape)  # used for calculating

        self.avg_pool = tf.keras.layers.AveragePooling2D(
            pool_size=(input_shape[-3], input_shape[-2]))

        self.conv1 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1, use_bias=True)

        self.conv2 = AtrousSpatialPyramidPooling._get_conv_block(
            kernel_size=1, dilation_rate=1)

        dummy_tensor = self.conv1(self.avg_pool(dummy_tensor))

        self.pool = tf.keras.layers.UpSampling2D(
            size=(
                input_shape[-3] // dummy_tensor.shape[1],
                input_shape[-2] // dummy_tensor.shape[2]
            ),
            interpolation='bilinear'
        )

        self.out1, self.out6, self.out12, self.out18 = map(
            lambda tup: AtrousSpatialPyramidPooling._get_conv_block(
                kernel_size=tup[0], dilation_rate=tup[1]
            ),
            [(1, 1), (3, 6), (3, 12), (3, 18)]
        )

    @staticmethod
    def _get_conv_block(kernel_size, dilation_rate, use_bias=False):
        return ConvBlock(256,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate,
                         padding='same',
                         use_bias=use_bias,
                         kernel_initializer=tf.keras.initializers.he_normal())


    def call(self, inputs, **kwargs):
        tensor = self.avg_pool(inputs)
        tensor = self.conv1(tensor)
        tensor = tf.keras.layers.Concatenate(axis=-1)([
            self.pool(tensor),
            self.out1(inputs),
            self.out6(inputs),
            self.out12(
                inputs
            ),
            self.out18(
                inputs
            )
        ])
        tensor = self.conv2(tensor)
        return tensor


class DeeplabV3Plus(tf.keras.Model):
    """DeeplabV3+ network architecture provider tf.keras.Model implementation.
    Args:
        num_classes:
            number of segmentation classes, effectively - number of output
            filters
        height, width:
            expected height, width of image
        backbone:
            backbone to be used
    """
    def __init__(self, input_shape, backbone='resnet50', **kwargs):
        super(DeeplabV3Plus, self).__init__()

        dummy_tensor = tf.random.normal(shape=(1, input_shape[0], input_shape[1], input_shape[2]))  # used for calculating

        self.n_channels = input_shape[-1]
        self.backbone = backbone
        
        self.input_layer = tf.keras.layers.Input(input_shape)
       
       # First conv layer
        self.init_conv = DeeplabV3Plus._get_conv_block(filters=3, 
                                                       kernel_size=3,
                                                       conv_activation='lrelu')
        dummy_tensor = self.init_conv(dummy_tensor)

        # Backbone feature 1
        self.backbone_feature_1 = self._get_backbone_feature('feature_1',
                                                             input_shape)
        dummy_tensor = self.backbone_feature_1(dummy_tensor)

        # Atrous spatial pyramidal pooling
        self.aspp = AtrousSpatialPyramidPooling(dummy_tensor.shape)

        self.input_a_upsampler_getter = self._get_upsample_layer_fn(
            input_shape, factor=4)

        # Backbone feature 2
        self.backbone_feature_2 = self._get_backbone_feature('feature_2',
                                                             input_shape)

        self.input_b_conv = DeeplabV3Plus._get_conv_block(filters=48,
                                                          kernel_size=(1, 1),
                                                          conv_activation="lrelu")

        self.conv1 = DeeplabV3Plus._get_conv_block(filters=256, 
                                                   kernel_size=3,
                                                   conv_activation='lrelu')

        self.conv2 = DeeplabV3Plus._get_conv_block(filters=256, 
                                                   kernel_size=3,
                                                   conv_activation='lrelu')

        self.otensor_upsampler_getter = self._get_upsample_layer_fn(
            input_shape, factor=1)

        self.out_conv = tf.keras.layers.Conv2D(filters=self.n_channels,
                                               kernel_size=(1, 1),
                                               padding='same')
        

        self.out = self.call(self.input_layer)


    @staticmethod
    def _get_conv_block(filters, kernel_size, conv_activation=None):
        return ConvBlock(filters, kernel_size=kernel_size, padding='same',
                         conv_activation=conv_activation,
                         kernel_initializer=tf.keras.initializers.he_normal(),
                         use_bias=False, dilation_rate=1)

    @staticmethod
    def _get_upsample_layer_fn(input_shape, factor: int):
        return lambda fan_in_shape: \
            tf.keras.layers.UpSampling2D(
                size=(
                    input_shape[0]
                    // factor // fan_in_shape[1],
                    input_shape[1]
                    // factor // fan_in_shape[2]
            ),
            interpolation="bilinear"
        )


    def _get_backbone_feature(self, feature: str,
                              input_shape) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=(input_shape[0], input_shape[1], 3))

        backbone_model = BACKBONES[self.backbone]['model'](
            input_tensor=input_layer, weights="imagenet", include_top=False) # weights='imagenet'

        output_layer = backbone_model.get_layer(
            BACKBONES[self.backbone][feature]).output
        return tf.keras.Model(inputs=input_layer, outputs=output_layer)


    def call(self, inputs, training=None, mask=None):

        inputs = self.init_conv(inputs)
        
        input_a = self.backbone_feature_1(inputs)

        input_a = self.aspp(input_a)
        input_a = self.input_a_upsampler_getter(input_a.shape)(input_a)

        input_b = self.backbone_feature_2(inputs)
        input_b = self.input_b_conv(input_b)

        tensor = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
        tensor = self.conv2(self.conv1(tensor))

        tensor = self.otensor_upsampler_getter(tensor.shape)(tensor)
        return self.out_conv(tensor)


if __name__ == "__main__":
    height = 64
    width = 128
    n_channels = 2
    x = tf.random.uniform(shape=(1, height, width, n_channels))
    input_shape = (height, width, n_channels)
    model = DeeplabV3Plus(input_shape=(height, width, n_channels))
    print("input: ", x.shape)
    y = model(x)
    print("output: ", y.shape)
    batch_size = 16
    print(model.summary())
    #get_model_memory_usage(batch_size, model, height, width, n_channels)(inputs)



