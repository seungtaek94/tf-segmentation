import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def display_sample(display_list, epoch, plot_dir):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """

    plt.figure(figsize=(10, 3))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i == 0:
            plt.imshow(display_list[i].numpy())
        elif i == 1:
            uniques, idx, counts = get_uniques(display_list[i])
            tf.print("MASK_Uniques =",uniques)
            plt.imshow(display_list[i].numpy())
        else:
            uniques, idx, counts = get_uniques(display_list[i])
            tf.print("Predict_Uniques =",uniques)

            plt.imshow(display_list[i].numpy())
            #print(display_list[i].numpy().shape)
            #print('@@@@@@@@@@@@@@@@@@@@@@@',np.count_nonzero(display_list[i].numpy()))
            #print(np.squeeze(display_list[i].numpy(), axis=-1).shape)
            #np.savetxt('sample1.csv', np.squeeze(display_list[i].numpy(), axis=-1), delimiter=",")
            #cv2.imwrite("test.png", display_list[i].numpy())

        plt.axis('off')
    plt.savefig(f'{plot_dir}/{epoch:0>3}.png', dpi=300)
    plt.show()


def display_sample_one(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """

    plt.figure(figsize=(10, 3))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if i == 0:
            plt.imshow(display_list[i].numpy())
        elif i == 1:
            plt.imshow(display_list[i].numpy())
        else:
            plt.imshow(display_list[i].numpy())
        plt.axis('off')

    plt.show()

def get_uniques(t):
    t1d = tf.reshape(t, shape=(-1,))
    # or tf.unique, if you don't need counts
    uniques, idx, counts = tf.unique_with_counts(t1d)
    return uniques, tf.reshape(idx, shape=tf.shape(t)), counts

    '''
    uniques, idx, counts = get_uniques(mask)
    tf.print("tf.shape(uniques) =", tf.shape(uniques))
    tf.print("tf.shape(idx) =", tf.shape(idx))
    tf.print("tf.shape(counts) =", tf.shape(counts))
    tf.print("uniques =", uniques)
    '''