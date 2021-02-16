import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

class CBA(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, dilation_rate=1, groups=1):
        super(CBA, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           dilation_rate=dilation_rate,
                                           strides=strides,
                                           padding="same")

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class ASPP(tf.keras.Model):
    def __init__(self, output_stride=32, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.ouput_stride = output_stride

        self.conv_1x1 = CBA(filters=256, kernel_size=1, strides=1)
        self.conv_3x3_1 = CBA(filters=256, kernel_size=3, strides=1, dilation_rate=atrous_rates[0])
        self.conv_3x3_2 = CBA(filters=256, kernel_size=3, strides=1, dilation_rate=atrous_rates[1])
        self.conv_3x3_3 = CBA(filters=256, kernel_size=3, strides=1, dilation_rate=atrous_rates[2])

        self.conv_1x1_out = CBA(filters=256, kernel_size=1, strides=1)

    def call(self, x, training=False):
        conv_1x1 = self.conv_1x1(x)
        conv_3x3_1 = self.conv_3x3_1(x)
        conv_3x3_2 = self.conv_3x3_2(x)
        conv_3x3_3 = self.conv_3x3_3(x)

        img_pooling = tf.reduce_mean(x, [1, 2], name='global_average_pooling', keepdims=True)
        img_pooling = tf.keras.layers.UpSampling2D(size=(H//self.ouput_stride, W//self.ouput_stride),
                                                   interpolation='bilinear')(img_pooling)

        out = tf.keras.layers.concatenate([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, img_pooling], axis=-1)
        out = tf.nn.leaky_relu(out)
        out = self.conv_1x1_out(out)

        return out # (N, 32, 64, 256)


class Decoder(tf.keras.Model):
    def __init__(self, identiy_1, identiy_2):
        super(Decoder, self).__init__()
        # input = (2, 32, 64, 256)

        self.identity_1 = identiy_1
        self.identity_2 = identiy_2

        #1/32
        self.up1_conv_0 = CBA(filters=256, kernel_size=1, strides=1)

        # 1/16
        self.up2 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up2_conv_0 = CBA(filters=256, kernel_size=1, strides=1)

        #1/8
        self.up3 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up3_conv_0 = CBA(filters=256, kernel_size=5, strides=1)
        self.up3_concat = tf.keras.layers.Concatenate(axis=-1)

        #1/4
        self.up4 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up4_conv_0 = CBA(filters=256, kernel_size=5, strides=1)
        self.up4_concat = tf.keras.layers.Concatenate(axis=-1)

        # 1/2
        self.up5 = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')
        self.up5_conv_1 = CBA(filters=256, kernel_size=5, strides=1)
        self.up5_conv_2 = CBA(filters=256, kernel_size=5, strides=1)
        self.up5_conv_3 = CBA(filters=3, kernel_size=1, strides=1)


    def call(self, x, training=False):
        # 1/32
        x = self.up1_conv_0(x)

        # 1/16
        x = self.up2(x)
        x = self.up2_conv_0(x)

        # 1/8
        x = self.up3(x)
        x = self.up3_conv_0(x)
        x = self.up3_concat([x, self.identity_1])

        # 1/4
        x = self.up4(x)
        x = self.up4_conv_0(x)
        x = self.up4_concat([x, self.identity_2])

        # 1/2
        x = self.up5(x)
        x = self.up5_conv_1(x)
        x = self.up5_conv_2(x)
        x = self.up5_conv_3(x)

        return x # size = (None, 512, 1024, 3)


class DeepLabV3(tf.keras.Model):
    def __init__(self, img_hight=1024, img_width=2048, encoder='resnet50v2', num_classes=3):
        super(DeepLabV3, self).__init__()

        global H, W
        H = img_hight
        W = img_width

        if encoder == 'resnet50v2':
            self.encoder = tf.keras.applications.ResNet50V2(input_shape=(H, W, 3), include_top=False)
            self.output_stride = 32
            self.identity_1 = self.encoder.get_layer(name='conv3_block4_1_relu').output # size = (None, 128, 256, 128)
            self.identity_2 = self.encoder.get_layer(name='conv2_block3_1_relu').output # size = (None, 256, 512, 64)

        self.aspp = ASPP(output_stride = self.output_stride, atrous_rates=[6, 12, 18])
        self.decoder = Decoder(self.identity_1, self.identity_2)


    def call(self, x, training=False):
        print('input: ', x.shape)
        x = self.encoder(x)
        print('encoder: ', x.shape)
        x = self.aspp(x)
        print('aspp: ', x.shape)
        x = self.decoder(x)
        print('out: ', x.shape)

        return x

if __name__ == "__main__":
    model = DeepLabV3(encoder='resnet50v2', num_classes=3)
    x = rand = tf.random.uniform([1, 1024, 2048, 3], 0, 1)
    y = model(x)