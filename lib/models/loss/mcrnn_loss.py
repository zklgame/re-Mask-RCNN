import tensorflow as tf
import keras.backend as K

from lib.models.loss.base_loss import smooth_l1_loss


def mrcnn_class_loss_graph(target_class_ids, pred_class_logits,
                           active_class_ids):
    '''
    Loss for the classifier head of Mask RCNN.

    :param target_class_ids:    [batch, num_rois] Integer class IDs. Uses zero padding to fill in the array.
    :param pred_class_logits:   [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
    :param active_class_ids:    [batch, num_classes] Has a value of 1 for classes that are in the dataset of the image,
                                and 0 for classes that are not in the dataset.
    :return: mrcnn_class_loss, cross_entropy loss with active class
    '''

    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Find predictions of classes that are not in the dataset.
    pred_class_ids = tf.argmax(pred_class_logits, axis=2)
    # TODO: Update this line to work with batch > 1. Right now it assumes all
    #       images in a batch have the same active_class_ids
    pred_active = tf.gather(active_class_ids[0], pred_class_ids)

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image.
    loss = loss * pred_active

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
    return loss

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    '''
    Loss for Mask R-CNN bounding box refinement.

    :param target_bbox:         [batch, num_rois, (dy, dx, log(dh), log(dw)] with zero padding
    :param target_class_ids:    [batch, num_rois] Integer class IDs. Uses zero padding to fill in the array with zero padding
    :param pred_bbox:           [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to proposal boxes
    :return: mrcnn_bbox_loss, smooth_l1_loss
    '''

    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 4))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = K.switch(tf.size(target_bbox) > 0,
                    smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    '''
    Mask binary cross-entropy loss for the masks head.

    :param target_masks:        [batch, num_rois, height = MASK_SHAPE[0] = MASK_POOL_SIZE * 2, width = MASK_SHAPE[1] = MASK_POOL_SIZE * 2]
                                A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    :param target_class_ids:    [batch, num_rois] Integer class IDs. Uses zero padding to fill in the array with zero padding
    :param pred_masks:          [batch, num_rois, MASK_POOL_SIZE * 2, MASK_POOL_SIZE * 2, NUM_CLASSES] float32 tensor with values from 0 to 1.
    :return: mrcnn_mask_loss, binary_crossentropy
    '''

    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss
