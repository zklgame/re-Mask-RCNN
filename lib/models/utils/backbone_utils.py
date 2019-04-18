import numpy as np
import math

from lib.models.utils.check_utils import _check_resnet


def compute_backbone_shapes(config, image_shape):
    '''
    Computes the width and height of each stage of the backbone network.
    :param config: instance of (sub-class of Config)
    :param image_shape: (h, w)
    :return: [N, (height, width)]. Where N is the number of stages
    '''

    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    _check_resnet(config.BACKBONE)

    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])
