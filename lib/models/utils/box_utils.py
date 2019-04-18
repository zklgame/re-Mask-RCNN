import tensorflow as tf
import numpy as np


def norm_boxes(boxes, shape):
    '''
    Converts boxes from pixel coordinates to normalized coordinates.
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    :param boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    :param shape: (height, width) in pixels
    :return: [N, (y1, x1, y2, x2)] in normalized coordinates
    '''

    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def norm_boxes_graph(boxes, shape):
    '''
    Converts boxes from pixel coordinates to normalized coordinates.
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    :param boxes: [B, None, (y1, x1, y2, x2)] in pixel coordinates
    :param shape: (height, width, ) in pixels
    :return: [B, None, (y1, x1, y2, x2)] in normalized coordinates
    '''

    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes(boxes, shape):
    '''
    Converts boxes from normalized coordinates to pixel coordinates.
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    :param boxes:   [..., (y1, x1, y2, x2)] in normalized coordinates
    :param shape:   (height, width) in pixels
    :return:        [..., (y1, x1, y2, x2)] in pixel coordinates
    '''

    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


def denorm_boxes_graph(boxes, shape):
    '''
    Converts boxes from normalized coordinates to pixel coordinates.
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    :param boxes:   [..., (y1, x1, y2, x2)] in normalized coordinates
    :param shape:   [..., (height, width)] in pixels
    :return:        [..., (y1, x1, y2, x2)] in pixel coordinates
    '''

    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)


def extract_bboxes(mask):
    '''
    Compute bounding boxes from masks.

    :param mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    :return:  bbox array [num_instances, (y1, x1, y2, x2)].
    '''

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    '''
    Calculates IoU of the given box with the array of the given boxes.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.

    :param box:         1D vector [y1, x1, y2, x2]
    :param boxes:       [boxes_count, (y1, x1, y2, x2)]
    :param box_area:    float. the area of 'box'
    :param boxes_area:  array of length boxes_count
    :return:
    '''

    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    '''
    Computes IoU overlaps between two sets of boxes.
    For better performance, pass the largest set first and the smaller second.

    :param boxes1:  [N1, (y1, x1, y2, x2)]
    :param boxes2:  [N2, (y1, x1, y2, x2)]
    :return: iou of shape [N1, N2]
    '''

    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

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
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)
