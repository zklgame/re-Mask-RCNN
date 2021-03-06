#  Feature Pyramid Network Heads

import keras.backend as K
import keras.layers as KL

from lib.models.roi_align.pyramid_roi_align import PyramidROIAlign
from lib.models.bn.batch_norm import BatchNorm


def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    '''
    Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    Algorithm:
        1. ROIAlign:
            Assign level(2-5) to rois
            for p in [P2, P3, P4, P5]:
                cropped_map = crop_and_resize level_roi from p
            return cropped_maps
        2. use cropped_maps to compute shared Conv map
        3. predict logits, probability and bbox_delta based on cropped_maps

    :param rois: [batch, num_rois, (y1, x1, y2, x2)] in normalized coordinates
    :param feature_maps: [P2, P3, P4, P5]
    :param image_meta:  [batch, config.IMAGE_META_SIZE] Image details. See compose_image_meta()
    :param pool_size:   int, e.g., 7. The width of the square feature map generated from ROI Pooling.
    :param num_classes: int, e.g., 11
    :param train_bn:    Boolean. Train or freeze Batch Norm layers
    :param fc_layers_size: int, e.g., 1024. Size of the 2 FC layers
    :return:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    '''

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # [batch, num_rois, fc_layers_size]
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    '''
    Builds the computation graph of the mask head of Feature Pyramid Network.

    Algorithm:
        1. ROIAlign:
            Assign level(2-5) to rois
            for p in [P2, P3, P4, P5]:
                cropped_map = crop_and_resize level_roi from p
            return cropped_maps
        2. use cropped_maps to compute sigmoid mask

    :param rois: [batch, num_rois, (y1, x1, y2, x2)] in normalized coordinates
    :param feature_maps: [P2, P3, P4, P5]
    :param image_meta:  [batch, config.IMAGE_META_SIZE] Image details. See compose_image_meta()
    :param pool_size:   int, e.g., 14. The width of the square feature map generated from ROI Pooling.
    :param num_classes: int, e.g., 11
    :param train_bn:    Boolean. Train or freeze Batch Norm layers
    :return: Masks [batch, num_rois, MASK_POOL_SIZE * 2, MASK_POOL_SIZE * 2, NUM_CLASSES]
    '''

    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x

