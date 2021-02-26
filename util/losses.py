import tensorflow as tf

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
    elif loss == 'focal':
        return focal_loss()
    elif loss == 'BCE':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss == 'dice':
        return dice_loss
    elif loss == 'soft_dice':
        return soft_dice_loss(smooth=0.2)
    else:
        print(f'\'{loss}\' is not 정의되지 않은 identify')
        return 0

def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  denominator = tf.reduce_sum(y_true + y_pred)

  return 1 - numerator / denominator

def soft_dice_loss(smooth=1.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred) + smooth
        denominator = tf.reduce_sum(y_true + y_pred) + smooth

        return 1 - numerator / denominator

    return loss



def ce_dice_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
      y_pred = tf.math.sigmoid(y_pred)
      numerator = 2 * tf.reduce_sum(y_true * y_pred)
      denominator = tf.reduce_sum(y_true + y_pred)

      return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores
# implemented by E. Moebel, 06/04/18
def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = tf.reduce_sum(p0 * g0, (0, 1, 2))
    den = num + alpha * tf.reduce_sum(p0 * g1, (0, 1, 2)) + beta * tf.reduce_sum(p1 * g0, (0, 1, 2))

    T = tf.reduce_sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = tf.cast(tf.shape(y_true)[-1], tf.float32)
    return Ncl - T