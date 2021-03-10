import tensorflow as tf
import tensorflow.keras.backend as K

def get_loss(loss:str = 'SCCE') -> tf.Tensor:
    '''
    SCCE        : Sparse Categorical Cross-Entropy  |   multi-class
    Focal       : Focal loss                        |   multi-class
    BCE         : Binary Cross-Entropy              |   Binary
    dice        : Dice loss                         |   Binary
    soft_dice   : Soft Dice loss                    |   Binary
    '''

    print('Loss: ', loss)

    if loss == 'SCCE':
        return tf.keras.losses.SparseCategoricalCrossentropy()
    elif loss == 'BCE':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss == 'dice':
        return dice_score_loss
    else:
        print(f'\'{loss}\' is not 정의되지 않은 identify')
        return 0

@tf.function
def dice_score_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int64)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

