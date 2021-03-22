import tensorflow as tf
from tensorflow.keras.initializers import random_normal
from tensorflow.python.keras.utils import data_utils

from .residual_block import Bottleneck, WEIGHTS_HASHES, BASE_WEIGHTS_PATH


class ResNetBackbone(tf.keras.layers.Layer):
    def __init__(self, resnet_type):
        resnet_spec = {50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50v2'),
                       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101v2'),
                       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152v2')}

        block, layers, channels, name = resnet_spec[resnet_type]

        self.net_name = name
        self.inplanes = 64

        super(ResNetBackbone, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same",
                                            use_bias=False, kernel_initializer=random_normal(mean=0, stddev=0.001))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")
        self.layer1 = self._make_layer(block, 64, blocks=layers[0])
        self.layer2 = self._make_layer(block, 128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks=layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = tf.keras.Sequential()
        layers.add(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.add(block(self.inplanes, planes))

        return layers

    def call(self, inputs, **kwargs):
        training = False
        if len(inputs) == 2:
            x, training = inputs
        else:
            x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        return x

    def init_weights(self):
        file_name = self.name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
        file_hash = WEIGHTS_HASHES[self.net_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        self.load_weights(weights_path)
        print("Initialize resnet from model zoo")
