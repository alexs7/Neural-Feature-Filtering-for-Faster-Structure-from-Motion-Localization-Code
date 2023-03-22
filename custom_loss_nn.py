import tensorflow as tf

@tf.function
def tweaked_loss(y_true, y_pred):
    # no rounding needed look at equation 3.14 in the paper

    #  At this point I create the tensor structure so that I can calculate the formula
    # from the paper. [1] -> [1,0] and [0] -> [0,1]. The left is positive class and the right is negative class
    # the 1 show which class it is each sample

    # This is where the reversing happens,
    # from paper: "the error of a certain sample is 0 if the sample is predicted correctly, otherwise the error is 1."
    y_pred = tf.concat([y_pred, 1.0 - y_pred], axis=1)
    y_true = tf.concat([y_true, 1.0 - y_true], axis=1)

    P_count = tf.reduce_sum(y_true[:,0])
    N_count = tf.reduce_sum(y_true[:,1])

    gt_p = tf.gather(y_true, tf.where(y_true[:,0] >= 0.5))
    pr_p = tf.gather(y_pred, tf.where(y_true[:,0] >= 0.5))

    # equation 3.5 in the paper
    # the inner sum
    FPE_inner_sum = tf.reduce_mean(tf.math.squared_difference(gt_p,pr_p), axis=-1)
    FPE = tf.reduce_sum(FPE_inner_sum) / P_count
    FPE = tf.where(tf.math.is_nan(FPE), 0., FPE)  # this is needed because if there are no P samples, FPE will be NaN

    gt_f = tf.gather(y_true, tf.where(y_true[:, 1] >= 0.5))
    pr_f = tf.gather(y_pred, tf.where(y_true[:, 1] >= 0.5))

    # equation 3.6 in the paper
    # the inner sum
    FNE_inner_sum = tf.reduce_mean(tf.math.squared_difference(gt_f, pr_f), axis=-1)
    FNE = tf.reduce_sum(FNE_inner_sum) / N_count
    FNE = tf.where(tf.math.is_nan(FNE), 0., FNE)  #this is needed because if there are no N samples, FNE will be NaN (this will most likely never happen)

    loss = (tf.math.pow((FPE + FNE), 2) + tf.math.pow((FPE - FNE), 2)) / 2

    return loss