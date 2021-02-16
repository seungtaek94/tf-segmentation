import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # ../datasets/cityscapes/leftImg8bit/train/*/*_leftImg8bit.png
    # Its corresponding annotation path is:
    # ../datasets/cityscapes/gtFine/train/*/*_gtFine_labelIds.png
    mask_path = tf.strings.regex_replace(img_path, "leftImg8bit", "gtFine")
    mask_path = tf.strings.regex_replace(mask_path, ".png", "_labelIds.png")
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)


    ''' 
    for c in ignore_class:
        mask = tf.where(mask == c, np.dtype('uint8').type(0), mask)
    '''
    mask = tf.where(mask != 7, np.dtype('uint8').type(0), mask)

    #plt.figure(figsize=(10, 10))
    #plt.imshow(mask, cmap='gray', vmin=0, vmax=10)
    #plt.show()

    return {'image': image, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict, img_hight=1024, img_width=2048) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.
    img_hight : int
    img_width : int

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (img_hight, img_width))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (img_hight, img_width))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict, img_hight=1024, img_width=2048) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.
    img_hight : int
    img_width : int

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (img_hight, img_width))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (img_hight, img_width))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def get_dataset(path: str, ignore_class: list, img_hight=1024, img_width=2048, batch_size=32) -> dict:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BUFFER_SIZE = 1000

    training_data_path = os.path.join(path, 'training/images')
    val_data_path = os.path.join(path, 'validation/images')

    training_data_dir = pathlib.Path(training_data_path)
    val_data_dir = pathlib.Path(val_data_path)

    TRAINSET_SIZE = len(list(training_data_dir.glob('*/*.png')))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

    VALSET_SIZE = len(list(val_data_dir.glob('*/*.png')))
    print(f"The Validation Dataset contains {VALSET_SIZE} images.")

    train_ds = tf.data.Dataset.list_files(str(training_data_path + '/*/*.png'), shuffle=False, seed=123)
    train_ds = train_ds.map(parse_image)

    val_ds = tf.data.Dataset.list_files(str(val_data_path + '/*/*.png'), shuffle=False, seed=123)
    val_ds = val_ds.map(parse_image)

    train_ds = train_ds.map(lambda x: load_image_train(x, img_hight, img_width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=BUFFER_SIZE, seed=123)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(lambda x: load_image_test(x, img_hight, img_width))
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return  {"train": train_ds, "val": val_ds}



if __name__ == '__main__':
    data_path = '/home/seungtaek/ssd1/datasets/mv'
    ignore_class = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, -1]

    dataset = get_dataset(data_path, ignore_class, 1024, 2048, batch_size=32)

    print("Train: ", dataset['train'])
    print("Val: ", dataset['val'])