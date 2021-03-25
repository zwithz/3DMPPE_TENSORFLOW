import tensorflow as tf
from tensorflow.keras.initializers import random_normal

BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
WEIGHTS_HASHES = {
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
}


class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels, 3, stride, padding="same", use_bias=False,
                                            kernel_initializer=random_normal(mean=0, stddev=0.001))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, 1, padding="same", use_bias=False,
                                            kernel_initializer=random_normal(mean=0, stddev=0.001))
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(self.expansion * out_channels, 1, stride, use_bias=False,
                                                       kernel_initializer=random_normal(mean=0, stddev=0.001)))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = None

    def call(self, inputs, training=None, **kwargs):
        identity = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        return tf.nn.relu(tf.keras.layers.add([identity, x]))


class Bottleneck(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels, 1, 1, padding="same", use_bias=False,
                                            kernel_initializer=random_normal(mean=0, stddev=0.001))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(out_channels, 3, stride, padding="same", use_bias=False,
                                            kernel_initializer=random_normal(mean=0, stddev=0.001))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(out_channels * self.expansion, 1, 1, padding="same", use_bias=False,
                                            kernel_initializer=random_normal(mean=0, stddev=0.001))
        self.bn3 = tf.keras.layers.BatchNormalization()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(
                tf.keras.layers.Conv2D(out_channels * self.expansion, 1, stride, use_bias=False,
                                       kernel_initializer=random_normal(mean=0, stddev=0.001)))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = None

    def call(self, inputs, training=None, **kwargs):
        identity = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        return tf.nn.relu(tf.keras.layers.add([identity, x]))
