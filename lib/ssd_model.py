# coding:utf-8
# __user__ = hiicy redldw
# __time__ = 2019/7/17
# __file__ = ssd_model
# __desc__ =
import tensorflow as tf


class SSD_(object):
    # FIXME：后面有空再改batch_size
    def __init__(self, n_class, batch_size, im_shape, device="/cpu:0"):
        self.n_class = n_class
        self.im_shape = im_shape
        self.feature_layer = ['conv4_3', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11']
        self.n_anchors = [4, 6, 6, 6, 4, 4]
        self.end_point = {}
        self.device = device
        self.loc_preds = []
        self.cls_preds = []
        self.batch_size = batch_size
        self.inputs = tf.placeholder("float32", shape=(batch_size, self.im_shape,
                                                       self.im_shape, 3), name="inputs")

    def _l2norm(self, x: tf.Tensor, scale, trainable=True, scope="L2Normalization"):
        n_channel = x.get_shape().as_list()[-1]
        l2_norm = tf.nn.l2_normalize(x, axis=[3], epsilon=1e-12)
        with tf.variable_scope(scope):
            # 不可自动处理错误获取变量，可用于全局共享变量
            gamma = tf.get_variable('gamma', shape=[n_channel, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(scale),
                                    trainable=trainable)
            return l2_norm * gamma

    def _detections(self, n_ft_map: tf.Tensor, n_class, n_anchor=2, scope=""):
        im_size = n_ft_map.get_shape().as_list()[0]
        with tf.name_scope("detection{}".format(n_anchor)):
            c = tf.layers.Conv2D(n_class * n_anchor, (3, 3), padding="same",
                                 name=f"{scope}_classify")(n_ft_map)
            r: tf.Tensor = tf.layers.Conv2D(4 * n_anchor, (3, 3), padding="same",
                                            name=f"{scope}_regress")(n_ft_map)
        loc_pred = tf.reshape(r, (-1, 4))  # 每个框的坐标
        cls_pred = tf.reshape(c, (-1, n_class))
        return loc_pred, cls_pred

    def build(self):
        with tf.device(self.device):
            with tf.name_scope("block1"):
                x = tf.layers.Conv2D(64, (3, 3), padding="same", name="conv1_1", activation="relu")(self.inputs)
                x = tf.layers.Conv2D(64, (3, 3), padding="same", name="conv1_2", activation="relu")(x)
                # 150*150*64
                x = tf.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)
            with tf.name_scope("block2"):
                x = tf.layers.Conv2D(128, (3, 3), padding="same",
                                     name="conv2_1", activation='relu')(x)
                x = tf.layers.Conv2D(128, (3, 3), padding="same",
                                     name="conv2_2", activation='relu')(x)
                # 75*75*128
                x = tf.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(x)
            with tf.name_scope("block3"):
                x = tf.layers.Conv2D(256, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv3_1')(x)
                x = tf.layers.Conv2D(256, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv3_2')(x)
                x = tf.layers.Conv2D(256, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv3_3')(x)
                # FIXME:为什么我需要pad
                x = tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
                # 下面执行完返回38*38*256
                x = tf.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(x)

            with tf.name_scope("block4"):
                # Block 4
                x = tf.layers.Conv2D(512, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv4_1')(x)
                x = tf.layers.Conv2D(512, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv4_2')(x)
                x = tf.layers.Conv2D(512, (3, 3),
                                     activation='relu',
                                     padding='same',
                                     name='conv4_3')(x)
                conv4_3 = self._l2norm(x, 0.1)
                self.end_point['conv4_3'] = conv4_3
                # 19*19*512
                x = tf.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
        with tf.name_scope("block5"):
            x = tf.layers.Conv2D(512, (3, 3), padding="same",
                                 activation="relu", name="conv5_1")(x)
            x = tf.layers.Conv2D(512, (3, 3), padding="same",
                                 activation="relu", name="conv5_2")(x)
            x = tf.layers.Conv2D(512, (3, 3), padding="same",
                                 activation="relu", name="conv5_3")(x)
            # 对比原来vgg这里有修改(19*19*512
            x = tf.layers.MaxPooling2D((3, 3), strides=(1, 1), padding="same", name="pool5")(x)
        x = tf.layers.Conv2D(1024, (3, 3), dilation_rate=6, padding="same",  # 关于扩张卷积
                             activation="relu", name="conv6")(x)  # 19*19*512

        x = tf.layers.Conv2D(1024, (1, 1), padding="same", activation="relu", name="conv7")(x)
        self.end_point['conv7'] = x
        # 四个维度
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        # (1,1)卷积一般只升、降维
        x = tf.layers.Conv2D(256, (1, 1), strides=(1, 1), activation="relu", name="conv8_1")(x)
        x = tf.layers.Conv2D(512, (3, 3), padding="valid", strides=(2, 2), activation="relu", name="conv8_2")(
            x)  # 10*10*512
        self.end_point['conv8'] = x

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.layers.Conv2D(128, (1, 1), activation="relu", name="conv9_1")(x)
        x = tf.layers.Conv2D(256, (3, 3), padding="valid", strides=(2, 2), activation="relu", name="conv9_2")(
            x)  # 5*5*256
        self.end_point['conv9'] = x

        x = tf.layers.Conv2D(128, (1, 1), activation="relu", name="conv10_1")(x)
        x = tf.layers.Conv2D(256, (3, 3), padding="valid", activation="relu", name="conv10_2")(x)  # 3*3*256
        self.end_point['conv10'] = x
        x = tf.layers.Conv2D(128, (1, 1), activation="relu", name="conv10_1")(x)
        x = tf.layers.Conv2D(256, (3, 3), activation="relu", name="conv10_2")(x)  # 1*1*256
        self.end_point['conv11'] = x
        for i in range(len(self.feature_layer)):
            loc_pred, cls_pred = self._detections(self.end_point[self.feature_layer[i]],
                                                  self.n_class,
                                                  self.n_anchors[i],
                                                  scope=self.feature_layer[i])
            self.loc_preds.append(loc_pred)
            self.cls_preds.append(cls_pred)  # 每个尺度的特征图
        # 叠加
        self.loc_preds = tf.concat(self.loc_preds, axis=0)
        self.cls_preds = tf.concat(self.cls_preds, axis=0)
        self.loc_preds = tf.reshape(self.loc_preds, [self.batch_size,
                                                     -1,
                                                     4])
        self.cls_preds = tf.reshape(self.cls_preds, [self.batch_size,
                                                     -1,
                                                     self.n_class])
        return self.loc_preds, self.cls_preds  # (-1,4),(-1,20)
