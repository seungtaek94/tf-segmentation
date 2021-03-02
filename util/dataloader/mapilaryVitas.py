import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import json
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import namedtuple
from util.func import *


Label = namedtuple('Label', [
    'name',
    'color',
    'id',
    'evaluate',
])

def pars_json_label(json_path: str) -> list:
    Labels = []
    with open(json_path) as json_file:
        json_data = json.load(json_file)

        labels = json_data['labels']
        for id, label in enumerate(labels):
            Labels.append(Label(label.get('readable'), label.get('color'), id, label.get('evaluate')))

    return Labels


def parse_image(img_path: str, ignore_class: list, img_height=512, img_width=1024) -> dict:
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (img_height, img_width))
    image = tf.cast(image, tf.uint8)

    # For one Image path:
    # ../datasets/mapillary_vitas/training/images/*.jpg
    # Its corresponding annotation path is:
    # ../datasets/mapillary_vitas/training/v2.0/labels/*.png
    mask_path = tf.strings.regex_replace(img_path, "images", "v2.0/labels")
    mask_path = tf.strings.regex_replace(mask_path, ".jpg", ".png")
    mask_row = tf.io.read_file(mask_path)
    mask_row = tf.image.decode_png(mask_row, channels=3)
    mask_row = tf.image.resize(mask_row, (img_height, img_width))
    mask_row = tf.cast(mask_row, tf.uint8)

    mask = tf.where(tf.reduce_all(mask_row == (128, 64, 128), axis=2), np.dtype('uint8').type(1), 0) # 도로

    mask += tf.where(tf.reduce_all(mask_row == (250, 170, 26), axis=2), np.dtype('uint8').type(1), 0) # Arrow Left
    mask += tf.where(tf.reduce_all(mask_row == (250, 170, 25), axis=2), np.dtype('uint8').type(1), 0)  # Arrow Other
    mask += tf.where(tf.reduce_all(mask_row == (250, 170, 24), axis=2), np.dtype('uint8').type(1), 0)  # Arrow Right
    mask += tf.where(tf.reduce_all(mask_row == (250, 170, 22), axis=2), np.dtype('uint8').type(1), 0)  # Split Left or Straight
    mask += tf.where(tf.reduce_all(mask_row == (250, 170, 21), axis=2), np.dtype('uint8').type(1), 0)  # Split Right or Straight
    mask += tf.where(tf.reduce_all(mask_row == (250, 170, 20), axis=2), np.dtype('uint8').type(1), 0)  # Straight
    #_mask2 += tf.where(tf.reduce_all(mask == (100, 128, 160), axis=2), np.dtype('uint8').type(10), 250)  # Manhole
    #_mask2 += tf.where(tf.reduce_all(mask == (220, 128, 128), axis=2), np.dtype('uint8').type(10), 250)  # Catch basin

    mask += tf.where(tf.reduce_all(mask_row == (255, 255, 255), axis=2), np.dtype('uint8').type(1), 0)  # Lane, Cross_walk
    mask += tf.where(tf.reduce_all(mask_row == (250, 170, 29), axis=2), np.dtype('uint8').type(1), 0)  # Zigzag

    #mask += tf.where(tf.reduce_all(mask_row == (196, 196, 196), axis=2), np.dtype('uint8').type(3), 0)  # Curb

    #mask += tf.where(tf.reduce_all(mask_row == (128, 128, 128), axis=2), np.dtype('uint8').type(4), 0)  # Traffic Sign Frame
    #mask += tf.where(tf.reduce_all(mask_row == (192, 192, 192), axis=2), np.dtype('uint8').type(4), 0)  # Traffic Sign - Ambiguous, back
    #mask += tf.where(tf.reduce_all(mask_row == (220, 220, 0), axis=2), np.dtype('uint8').type(4), 0)  # Traffic Sign - Direction, temp
    #mask += tf.where(tf.reduce_all(mask_row == (0, 0, 196), axis=2), np.dtype('uint8').type(4), 0)    # Traffic Sign - Parking

    mask = tf.where(mask == 0, np.dtype('uint8').type(0), mask)

    mask = tf.expand_dims(mask, axis=-1)

    return {'image': image, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32)
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict, img_height=1024, img_width=2048) -> tuple:
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def get_dataset(path: str, ignore_class: list, img_height=512, img_width=1024, batch_size=32) -> dict:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BUFFER_SIZE = 1000

    training_data_path = os.path.join(path, 'training/images')
    val_data_path = os.path.join(path, 'validation/images')

    training_data_dir = pathlib.Path(training_data_path)
    val_data_dir = pathlib.Path(val_data_path)

    TRAINSET_SIZE = len(list(training_data_dir.glob('*.jpg')))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

    VALSET_SIZE = len(list(val_data_dir.glob('*.jpg')))
    print(f"The Validation Dataset contains {VALSET_SIZE} images.")

    train_ds = tf.data.Dataset.list_files(str(training_data_path + '/*.jpg'), shuffle=False, seed=123)
    train_ds = train_ds.map(lambda x: parse_image(x, ignore_class, img_height, img_width))

    val_ds = tf.data.Dataset.list_files(str(val_data_path + '/*.jpg'), shuffle=False, seed=123)
    val_ds = val_ds.map(lambda x: parse_image(x, ignore_class, img_height, img_width))


    train_ds = train_ds.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE, seed=123)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(lambda x: load_image_test(x, img_height, img_width))
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


    return  {"train": train_ds, "val": val_ds}



if __name__ == '__main__':
    data_path = '/home/seungtaek/ssd1/datasets/mapillary_vistas'
    json_path = os.path.join(data_path, 'config_v2.0.json')
    print(json_path)
    Labels = pars_json_label(json_path)


    ignore_class = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, -1]

    dataset = get_dataset(data_path, ignore_class, 512, 1024, batch_size=32)


    for i, data in enumerate(dataset['train']):
        display_list = []
        print(data[0][0].shape)
        print(data[1][0].shape)

        uniques, idx, counts = get_uniques(data[1][0])

        tf.print("tf.shape(uniques) =", tf.shape(uniques))
        tf.print("tf.shape(idx) =", tf.shape(idx))
        tf.print("tf.shape(counts) =", tf.shape(counts))
        tf.print("uniques =", uniques)

        display_list.append(data[0][0])
        display_list.append(data[1][0])
        display_sample(display_list)
        if i == 3: break

    display_sample(display_list)


    print("Train: ", dataset['train'])
    print("Val: ", dataset['val'])