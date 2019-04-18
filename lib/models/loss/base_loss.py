import keras.backend as K


def smooth_l1_loss(y_true, y_pred):
    '''
    Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    :param y_true:
    :param y_pred:
    :return:
    '''

    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss
