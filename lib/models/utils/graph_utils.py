import tensorflow as tf


def trim_zeros_graph(boxes, name='trim_zeros'):
    '''
    Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    :param boxes: [N, 4] matrix of boxes
    :param name: operation name
    :return:
        boxes: [K, 4]
        non_zeros: idx of non-zero boxes
    '''

    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def box_refinement_graph(box, gt_box):
    '''
    Compute refinement needed to transform box to gt_box.

    :param box: [N, (y1, x1, y2, x2)] in normalized coordinates.
    :param gt_box: [N, (y1, x1, y2, x2)] in normalized coordinates.
    :return: delta: [N, (dy, dx, log(dh), log(dw))]
    '''

    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def batch_pack_graph(x, counts, num_rows):
    '''
    Picks different number of values from each row in x depending on the values in counts.
    :param x:  [batch, max positive anchors, (dy, dx, log(dh), log(dw))]
    :param counts: [batch, ] of int
    :param num_rows: config.IMAGES_PER_GPU
    :return:
    '''

    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)
