# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

import keras.layers as KL

from lib.models.utils.check_utils import _check_resnet
from lib.models.bn.batch_norm import BatchNorm


def base_block(input_tensor, stage, block, filters, kernel_size,
               use_bias, train_bn, is_first_block=False, strides=(1, 1)):
    '''
    The base_block to construct identity_block and conv_block according to `is_first_block`

    :param input_tensor: [B, H, W, C]
    :param stage: integer, current stage label, used for generating layer names
    :param block: 'a','b'..., current block label, used for generating layer names
    :param filters: list of integers, the nb_filters of 3 conv layer at main path
    :param kernel_size: default 3, the kernel size of middle conv layer at main path
    :param use_bias: Boolean. To use or not use a bias in conv layers.
    :param train_bn: Boolean. Train or freeze Batch Norm layers
    :param is_first_block: Boolean. if True, construct conv_block, else construct identity_block
    :param strides: list of integers. Only use if for conv_block
    :return: [B, H', W', C']
    '''

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    strides = strides if is_first_block else (1, 1)

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    if is_first_block:
        shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
        shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)
    else:
        shortcut = input_tensor

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    '''
    The identity_block is the block that has no conv layer at shortcut
    see `base_block` for details
    '''

    return base_block(input_tensor, stage, block, filters, kernel_size, use_bias, train_bn, is_first_block=False)


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    '''
    conv_block is the block that has a conv layer at shortcut
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    see `base_block` for details
    '''

    return base_block(input_tensor, stage, block, filters, kernel_size, use_bias, train_bn, is_first_block=True, strides=strides)


def resnet(input_image, architecture, stage5=False, train_bn=True):
    '''
    Build a ResNet graph.

    :param input_image:     [B, H, W, C]
    :param architecture:    Can be 'resnet50' or 'resnet101'
    :param stage5:          Boolean. If False, stage5 of the network is not created
    :param train_bn:        Boolean. Train or freeze Batch Norm layers
    :return: [C1, C2, C3, C4, C5]
    '''

    _check_resnet(architecture)

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]
