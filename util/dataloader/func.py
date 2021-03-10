import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

@tf.function
def augmentation(input_image:tf.Tensor, input_mask: tf.Tensor) -> tuple:
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    rotate_ratio = 0.2

    if tf.random.uniform(()) > 0.5:
        angle = np.pi*2 - tf.random.uniform((), maxval=rotate_ratio)
    else:
        angle = tf.random.uniform((), maxval=rotate_ratio)

    input_image = tfa.image.rotate(input_image, angle)
    input_mask = tfa.image.rotate(input_mask, angle)

    if tf.random.uniform(()) > 0.8:
        input_image = tf.image.random_brightness(input_image, max_delta=0.2)  # Random brightness
        input_image = tf.image.random_hue(input_image, 0.1)
        input_image = tf.image.random_contrast(input_image, 0.4, 1)
        input_image = tf.image.random_saturation(input_image, 0.4, 1)
        input_image = tf.clip_by_value(input_image, 0, 1)

    return input_image, input_mask