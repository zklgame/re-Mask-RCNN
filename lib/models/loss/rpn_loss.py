import tensorflow as tf
import keras.backend as K

from lib.models.utils.graph_utils import batch_pack_graph
from lib.models.loss.base_loss import smooth_l1_loss


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    '''
    return RPN anchor classifier loss.

    :param rpn_match:           [batch, anchors, 1]. Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    :param rpn_class_logits:    [batch, anchors = H * W * anchors_per_location, 2] Anchor classifier logits BG/FG. (before softmax)
    :return: rpn class loss, Cross entropy loss
    '''

    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    '''
    return RPN bounding box loss

    :param config: instance of (sub-class of Config)
    :param target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed bbox deltas.
    :param rpn_match: [batch, anchors, 1] . Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
    :param rpn_bbox: proposaled [batch, anchors = H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors
    :return: rpn bbox loss, smooth_l1_loss
    '''
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                   config.IMAGES_PER_GPU)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss
