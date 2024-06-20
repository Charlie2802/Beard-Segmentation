from __future__ import division
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
alpha = 0.2
kernel = tf.zeros((15, 15, 1))

# @tf.function
def get_boundary(inp):
    dilated = tf.nn.dilation2d(inp, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), data_format="NHWC", padding="SAME")
    eroded = tf.nn.erosion2d(inp, filters=kernel, strides=(1,1,1,1), dilations=(1,1,1,1), data_format="NHWC", padding="SAME")
    diff = tf.abs(dilated - eroded)
    diff = tf.expand_dims(diff, axis=-1)
    return tf.abs(dilated-eroded)

@tf.function
def refine_loss(logits, image, boundary_target):
    
    gamma1 = 0.5
    gamma2 = 1-gamma1
    factor_lambda= 1.5

    dy_logits, dx_logits = tf.image.image_gradients(logits)
    dy_image, dx_image = tf.image.image_gradients(image)

    # magnitudes of logits and labels gradients
    Mpred = tf.sqrt(tf.square(dy_logits)+tf.square(dx_logits))
    Mimg = tf.sqrt(tf.square(dy_image)+tf.square(dx_image))

    # define cos loss and mag loss
    cosL = (1-tf.abs(dx_image*dx_logits+dy_image*dy_logits))*Mpred
    magL = tf.maximum(factor_lambda*Mimg-Mpred,0)

    # define mask
    M_bound = boundary_target

    # define total refine loss
    refineLoss = (gamma1*cosL + gamma2*magL)*M_bound
    return tf.reduce_mean(refineLoss)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score
    
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    
    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + \
    weighted_dice_loss(y_true, y_pred, weight)
    return loss

def loss_fn(image):
    def custom_loss(label, predictions):
        label_mask = tf.cast(label, tf.float32)
        pred_mask = tf.cast(predictions, tf.float32)

        label_boundary = get_boundary(label_mask)
        pred_boundary = get_boundary(pred_mask)

        rloss = refine_loss(pred_mask, image, label_boundary)
        #b_loss = binary_crossentropy(label, predictions)
        b_w_loss = weighted_bce_dice_loss(label, predictions)

        return  (alpha * rloss) + ((1 - alpha) * b_w_loss)
    return custom_loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   #strict=True,
                   name="loss"
                   )
    return loss

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def keras_lovasz_hinge(labels,logits):
    return lovasz_hinge(logits, labels, per_image=True, ignore=None)

def focal_loss(y_true, y_pred):
    gamma=0.75
    alpha=0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def boundary_loss(label, predictions):
    gt_boundary = get_boundary(label)
    pr_boundary = get_boundary(predictions)
#     b_loss = (0.7 * binary_crossentropy(gt_boundary, pr_boundary)) + (0.3 * focal_loss(gt_boundary, pr_boundary))
#     b_loss = (0.5 * binary_crossentropy(gt_boundary, pr_boundary)) + (0.5 * binary_focal_loss(gt_boundary, pr_boundary))
    return binary_focal_loss(gt_boundary, pr_boundary)

# def total_loss(y_true, y_pred):
# #     return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred) + keras_lovasz_hinge(y_true, y_pred) + bce_jaccard_loss(y_true, y_pred)
# #     return (0.2*binary_focal_loss(y_true, y_pred)) + (0.5*dice_loss(y_true, y_pred)) + (0.2*weighted_bce_dice_loss(y_true, y_pred)) + (0.1 * boundary_loss(y_true, y_pred))
# #     return (0.5*dice_loss(y_true, y_pred)) + (0.1*boundary_loss(y_true, y_pred)) + (0.2*binary_crossentropy(y_true, y_pred)) + (0.2*binary_focal_loss(y_true, y_pred))
# #     return (0.9 * dice_loss(y_true, y_pred)) + (0.1 * boundary_loss(y_true, y_pred)) #+ (0.3 * binary_crossentropy(y_true, y_pred))
#    return (0.3 * focal_loss(y_true, y_pred)) + (0.7 * binary_crossentropy(y_true, y_pred))

def total_loss(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
#     final_loss = focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred) + keras_lovasz_hinge(y_true, y_pred) + bce_jaccard_loss(y_true, y_pred)
    final_loss = (0.2*binary_focal_loss(y_true, y_pred)) + (0.5*dice_loss(y_true, y_pred)) + (0.2*weighted_bce_dice_loss(y_true, y_pred)) + (0.1 * boundary_loss(y_true, y_pred))
#     final_loss = (0.5*dice_loss(y_true, y_pred)) + (0.1*boundary_loss(y_true, y_pred)) + (0.2*binary_crossentropy(y_true_flat, y_pred_flat)) + (0.2*binary_focal_loss(y_true, y_pred))
#     final_loss = (0.9 * dice_loss(y_true, y_pred)) + (0.1 * boundary_loss(y_true, y_pred)) #+ (0.3 * binary_crossentropy(y_true_flat, y_pred_flat))
#     final_loss = (0.3 * focal_loss(y_true, y_pred)) + (0.7 * binary_crossentropy(y_true_flat, y_pred_flat))
#     tf.print("Shape of total loss5: ", final_loss.shape, "Value: ", final_loss)
    return final_loss

def IOU(y_true, y_pred):
    
    smooth = 1e-6
    y_true ,y_pred = tf.reshape(y_true, [-1]),tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_pred * y_true) + smooth
    union = tf.reduce_sum(y_pred+y_true) - inter + smooth
    return inter / union

def dice(y_true, y_pred):
    
    smooth = 1e-6
#     y_true ,y_pred = tf.reshape(y_true, [-1]),tf.reshape(y_pred, [-1])
    inter = 2.*tf.reduce_sum(y_pred * y_true) + smooth
    sum_ = tf.reduce_sum(y_pred+y_true) + smooth
    return inter / sum_

def hair_loss(y_true, y_pred):
    y_true ,y_pred = tf.reshape(y_true, [-1]),tf.reshape(y_pred, [-1])
#     y_pred = tf.math.sigmoid(y_pred)
    bce = binary_crossentropy(y_true, y_pred, from_logits=True)
    y_pred = tf.math.sigmoid(y_pred)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)
    d_loss = 1. - dice(y_true,y_pred)
    final_loss = (1 * bce) + (3 * d_loss)
#     tf.print("Shape of hair loss: ", final_loss.shape, "Value: ", final_loss)
    return final_loss

def matting_loss(image, mask):
    image = tf.image.rgb_to_grayscale(image)
    sobel_kernel_x = tf.constant([[1.0, 0.0, -1.0],
                                  [2.0, 0.0, -2.0],
                                  [1.0, 0.0, -1.0]], dtype=tf.float32)
    sobel_kernel_x = tf.expand_dims(sobel_kernel_x, -1)
    sobel_kernel_x = tf.expand_dims(sobel_kernel_x, -1)
    # print(sobel_kernel_x.shape)
    # print(image.shape)
    I_x = tf.nn.conv2d(image, sobel_kernel_x, strides=[1, 1, 1, 1], padding='SAME')
    M_x = tf.nn.conv2d(mask, sobel_kernel_x, strides=[1, 1, 1, 1], padding='SAME')

    sobel_kernel_y = tf.constant([[1.0, 2.0, 1.0],
                                  [0.0, 0.0, 0.0],
                                  [-1.0, -2.0, -1.0]], dtype=tf.float32)
    sobel_kernel_y = tf.expand_dims(sobel_kernel_y, -1)
    sobel_kernel_y = tf.expand_dims(sobel_kernel_y, -1)
    I_y = tf.nn.conv2d(image, sobel_kernel_y, 1, padding='SAME')
    M_y = tf.nn.conv2d(mask, sobel_kernel_y, 1, padding='SAME')

    # I_x = I_x / tf.math.reduce_max(I_x)
    # I_y = I_y / tf.math.reduce_max(I_y)
    # M_x = M_x / tf.math.reduce_max(M_x)
    # M_y = M_y / tf.math.reduce_max(M_y)

#     I_x, I_y = tf.image.image_gradients(image)
#     M_x, M_y = tf.image.image_gradients(mask)

    M_mag = tf.sqrt(tf.pow(M_x, 2) + tf.pow(M_y, 2) + 1e-8)
    temp = 1 - (tf.pow((I_x * M_x) + (I_y * M_y), 2))
    temp = tf.where(temp > 0, temp, 0.0)

    loss = (tf.reduce_sum(M_mag * temp) / tf.reduce_sum(M_mag)) + 1e-6
    return loss