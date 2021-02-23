import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2,2))

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

class Unet(tf.keras.Model):
    def __init__(self, img_height, img_width, classes_num=35):
        super(Unet, self).__init__()

        self.conv_down1 = DownSample(64)
        self.conv_down2 = DownSample(128)
        self.conv_down3 = DownSample(256)
        self.conv_down4 = DownSample(512)

        self.conv_down5 = DownSample(1024)

        self.conv_up1 = UpSample(512)
        self.conv_up2 = UpSample(256)
        self.conv_up3 = UpSample(128)
        self.conv_up4 = UpSample(64)

        if classes_num == 2:
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
        _, x = self.conv_down5(x)

        x = self.conv_up1([x, conv4])
        x = self.conv_up2([x, conv3])
        x = self.conv_up3([x, conv2])
        x = self.conv_up4([x, conv1])

        x = self.out(x)

        return x

if __name__ == "__main__":

    model = Unet(256, 512, 35)

    x = tf.random.uniform([2, 256, 512, 3], 0, 1)
    y = model(x)
    print(y.shape)

    model.summary()