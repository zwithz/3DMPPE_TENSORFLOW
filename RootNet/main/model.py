import tensorflow as tf
from tensorflow.keras.initializers import random_normal, constant
from common.nets.resnet import ResNetBackbone
from config import cfg


class RootNet(tf.keras.Model):

    def __int__(self):
        self.inplanes = 2048
        self.outplanes = 256

        super(RootNet, self).__int__()
        self.deconv_layers = self._make_deconv_layer(3)
        self.xy_layer = tf.keras.layers.Conv2D(1, 1, 1, padding='same')
        self.depth_layer = tf.keras.layers.Conv2D(1, 1, 1, padding='same')

    def _make_deconv_layer(self, num_layers):
        layers = tf.keras.Sequential()
        inplanes = self.inplanes
        outplanes = self.outplanes
        for i in range(num_layers):
            layers.add(tf.keras.layers.Conv2DTranspose(outplanes, 4, 2, padding='same', output_padding=0,
                                                       use_bias=False, input_shape=inplanes))
            layers.add(tf.keras.layers.BatchNormalization())
            layers.add(tf.nn.relu())
            inplanes = outplanes

        return layers

    def call(self, inputs, **kwargs):
        x, k_value = inputs

        # x, y
        xy = self.deconv_layers(x)
        xy = self.xy_layer(xy)

        # Soft-argmax
        xy = tf.reshape(xy, [-1, 1, cfg.output_shape[0] * cfg.output_shape[1]])
        xy = tf.nn.softmax(xy, 2)
        xy = tf.reshape(xy, [-1, 1, cfg.output_shape[0], cfg.output_shape[1]])

        hm_x = tf.reduce_sum(xy, 2)
        hm_y = tf.reduce_sum(xy, 3)

        # `SoftArgmax(x) = \sum_i(i * softmax(x)_i)`
        coord_x = hm_x * tf.range(cfg.output_shape[1], dtype=tf.float32)
        coord_y = hm_y * tf.range(cfg.output_shape[0], dtype=tf.float32)
        coord_x = tf.reduce_sum(coord_x, 2)
        coord_y = tf.reduce_sum(coord_y, 2)

        # global average pooling
        img_feat = tf.reduce_mean(tf.reshape(x, [x.size(0), x.size(1), x.size(2) * x.size(3)]), 2)
        img_feat = tf.expand_dims(img_feat, 2)
        img_feat = tf.expand_dims(img_feat, 3)

        gamma = self.deconv_layers(img_feat)
        gamma = tf.reshape(gamma, [-1, 1])
        depth = gamma * tf.reshape(k_value, [-1, 1])

        coord = tf.concat([coord_x, coord_y, depth], 2)
        return coord

    def init_weights(self):
        for name, m in self.deconv_layers.submodules():
            if isinstance(m, tf.keras.layers.Conv2DTranspose):
                m.kernel_initializer = random_normal(stddev=0.001)
            elif isinstance(m, tf.keras.layers.BatchNormalization):
                m.gamma_initializer = constant(1)
                m.beta_initializer = constant(0)
        for m in self.xy_layer.submodules():
            if isinstance(m, tf.keras.layers.Conv2D):
                m.kernel_initializer = random_normal(stddev=0.001)
                m.bias_initializer = constant(0)
        for m in self.depth_layer.submodules():
            if isinstance(m, tf.keras.layers.Conv2D):
                m.kernel_initializer = random_normal(stddev=0.001)
                m.bias_initializer = constant(0)


class ResPoseNet(tf.keras.Model):
    def __init__(self, backbone, root):
        super(ResPoseNet, self).__init__()
        self.backbone = backbone
        self.root = root

    def call(self, inputs, **kwargs):
        target = None
        if len(inputs) == 3:
            input_img, k_value, target = inputs
        else:
            input_img, k_value = inputs

        feature_map = self.backbone([input_img])
        coord = self.root([feature_map, k_value])

        if target is None:
            return coord
        else:
            target_coord = target['coord']
            target_vis = target['vis']
            target_have_depth = target['have_depth']

            tmp = tf.abs(coord - target_coord)
            tmp = tf.reshape(tmp, [3, -1])
            loss_coord = tmp * target_vis
            loss_coord = tf.reshape(loss_coord, [-1, 3])
            loss_coord = (loss_coord[:, 0] + loss_coord[:, 1] + loss_coord[:, 2]
                          * tf.reshape(target_have_depth, -1)) / 3.
            return loss_coord


def get_pose_net(cfg, is_train):
    backbone = ResNetBackbone(cfg.resnet_type)
    root_net = RootNet()
    if is_train:
        backbone.init_weights()
        root_net.init_weights()

    model = ResPoseNet(backbone, root_net)
    return model
