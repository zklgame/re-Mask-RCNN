import numpy as np


def generate_anchors(scale, ratios, shape, feature_stride, anchor_stride):
    '''

    :param scale: int, e.g., 32
    :param ratios: list, e.g., [0.5, 1, 2]
    :param shape: list, (height, width), spatial shape of the feature map over which
            to generate anchors.
    :param feature_stride: int, e.g., 4
    :param anchor_stride: int, e.g., 1
    :return: [anchors_num, (y1, x1, y2, x2)]
    '''

    # Get all combinations of scales and ratios
    scale, ratios = np.meshgrid(np.array(scale), np.array(ratios))
    scale = scale.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scale / np.sqrt(ratios)
    widths = scale * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    '''
    Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    :param scales: list, e.g., (32, 64, 128, 256, 512)
    :param ratios: list, e.g., [0.5, 1, 2]
    :param feature_shapes: [N, (height, width)]. Where N is the number of stages
    :param feature_strides: list, e.g., [4, 8, 16, 32, 64]
    :param anchor_stride: int, e.g., 1
    :return: anchors: [anchor_count, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    '''

    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)

