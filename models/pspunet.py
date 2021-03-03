import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


class Block(tf.keras.Model):
    def __init__(self, filters):
        super(Block, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, activation='relu', padding='same')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownSample(tf.keras.Model):
    def __init__(self, filters):
        super(DownSample, self).__init__()

        self.conv_block = Block(filters)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, x):
        x = self.conv_block(x)
        pool_x = self.pool(x)

        return pool_x, x


class UpSample(tf.keras.Model):
    def __init__(self, filters):
        super(UpSample, self).__init__()
        self.conv_up = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, activation='relu')
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.conv_block = Block(filters)

    def call(self, inputs):
        x1, x2 = inputs
        x1 = self.conv_up(x1)
        x1 = self.concat([x1, x2])
        x1 = self.conv_block(x1)

        return x1


class PyramidPoolingModule(tf.keras.Model):
    def __init__(self, H=64, W=128, filters=512):
        super(PyramidPoolingModule, self).__init__()
        self.input_h = H
        self.input_w = W
        self.input_filters = filters
        self.concat = tf.keras.layers.Concatenate(axis=-1)

        self.pooling_layer_1 = self.pooling_layer(1)
        self.pooling_layer_2 = self.pooling_layer(2)
        self.pooling_layer_3 = self.pooling_layer(3)
        self.pooling_layer_4 = self.pooling_layer(4)


    def call(self, x):
        p1 = self.pooling_layer_1(x)
        p1 = tf.raw_ops.ResizeBilinear(images=p1, size=(self.input_h, self.input_w))
        p2 = self.pooling_layer_2(x)
        p2 = tf.raw_ops.ResizeBilinear(images=p2, size=(self.input_h, self.input_w))
        p3 = self.pooling_layer_3(x)
        p3 = tf.raw_ops.ResizeBilinear(images=p3, size=(self.input_h, self.input_w))
        p4 = self.pooling_layer_4(x)
        p4 = tf.raw_ops.ResizeBilinear(images=p4, size=(self.input_h, self.input_w))

        x = self.concat([x, p1, p2, p3, p4])

        return x

    def pooling_layer(self, pool_ratio):
        layer = tf.keras.Sequential()
        layer.add(tf.keras.layers.AveragePooling2D((self.input_h//pool_ratio, self.input_w//pool_ratio)))
        layer.add(tf.keras.layers.Conv2D(self.input_filters//4, 1, padding='same', activation='relu'))
        layer.add(tf.keras.layers.BatchNormalization())
        return layer


class Unet(tf.keras.Model):
    def __init__(self, img_height, img_width, classes_num=35):
        super(Unet, self).__init__()

        self.conv_down1 = DownSample(64)
        self.conv_down2 = DownSample(128)
        self.conv_down3 = DownSample(256)
        self.conv_down4 = DownSample(512)

        self.pyramid_pool = PyramidPoolingModule()

        self.conv_up2 = UpSample(256)
        self.conv_up3 = UpSample(128)
        self.conv_up4 = UpSample(64)

        if classes_num == 1:
            self.out =  tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='Output')
            print('Out Put with Sigmoid')
        else:
            self.out =  tf.keras.layers.Conv2D(classes_num, 1, padding='same', activation='softmax', name='Output')
            print('Out Put with Softmax')

    def call(self, x):
        x, conv1 = self.conv_down1(x)
        x, conv2 = self.conv_down2(x)
        x, conv3 = self.conv_down3(x)
        x, conv4 = self.conv_down4(x)

        x = self.pyramid_pool(conv4)

        x = self.conv_up2([x, conv3])
        x = self.conv_up3([x, conv2])
        x = self.conv_up4([x, conv1])

        x = self.out(x)

        return x


if __name__ == "__main__":
    model = Unet(512, 1024, 5)

    x = tf.random.uniform([1, 512, 1024, 3], 0, 1)
    y = model(x)
    print(y.shape)

    model.summary()