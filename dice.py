import keras.backend as K
import numpy as np
import tensorflow as tf

smooth = 1.


def dice_coef_np(y_true, y_pred, threshold_mode=False):
    """dice计算
    y_true：真值
    y_pred:预测结果
    threshold_mode：True表示计算过程将y_pred设为概率最大的类别
    """
    num_classes = 6
    dice = 0.0
    dices = []
    if threshold_mode:
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(y_true, axis=-1)
        for i in range(1, num_classes):
            y_true_f = (y_true == i)
            y_pred_f = (y_pred == i)
            intersection = np.sum(y_true_f * y_pred_f)
            dices.append((2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))
            dice += (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    else:
        for i in range(1, num_classes):
            y_true_f = y_true[..., i].flatten()
            y_pred_f = y_pred[..., i].flatten()
            intersection = np.sum(y_true_f * y_pred_f)
            dices.append((2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))
            dice += (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice / (num_classes-1), dices


def dice_coef(y_true, y_pred):
    numerator = y_true * y_pred
    numerator = K.sum(numerator, (0, 1, 2, 3))
    denominator = y_true + y_pred
    denominator = K.sum(denominator, (0, 1, 2, 3))
    dice = (2. * numerator + smooth) / (denominator + smooth)
    return K.mean(dice)


def dice_coef_loss(classes_num, classes_w_t2=None):
    """dice loss计算
    classes_num：为一个list，表示每个类别的样本数量。classes_w_t2为None时，根据此计算每个类别的权重classes_w_t2
    classes_w_t2：不为None时，表示直接指定每个类别的权重
    """
    if classes_w_t2 is None:
        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
    print("classes_weights", classes_w_t2)
    classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=tf.float32)
    def inner(y_true, y_pred):
        numerator = y_true * y_pred
        numerator = K.sum(numerator, (0, 1, 2, 3))
        denominator = y_true + y_pred
        denominator = K.sum(denominator, (0, 1, 2, 3))
        dice = (2. * numerator + smooth) / (denominator + smooth)
        return 1. - K.sum(dice * classes_w_tensor)
    return inner

def focal_loss(alpha=0.5, gamma=2.0, num_classes=6):
    def inner(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        #pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -num_classes * K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))#-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return inner
