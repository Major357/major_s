from __future__ import division

import numpy as np
import six
import major_config


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = major_config.num_classes
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()   # (168960, )
        gt_label = gt_label.flatten()   # (168960, )

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.  极度巧妙 × class_nums 正好使得每个ij能够对应.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2)\
            .reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


# 图像分割的常用评测函数：https://blog.csdn.net/qq_41375318/article/details/108380694

# def generate_matrix(gt_image, pre_image, num_class=cfg.DATASET_Num_Class):
#     mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
#
#     label = num_class * gt_image[mask].astype('int') + pre_image[mask]
#     # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
#     count = np.bincount(label, minlength=num_class ** 2)
#     confusion_matrix = count.reshape(num_class, num_class)  # 21 * 21(for pascal)
#     return confusion_matrix

# PA
def Pixel_Accuracy(confusion_matrix):
    Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return Acc

# MPA
def Pixel_Accuracy_Class(confusion_matrix):
        Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

# MIoU
def Mean_Intersection_over_Union(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)  # 跳过0值求mean,shape:[21]
    return MIoU

# FWIoU
def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU


def eval_semantic_segmentation(pred_labels, gt_labels):
    confusion = calc_semantic_segmentation_confusion(pred_labels, gt_labels)
    pa = Pixel_Accuracy(confusion)
    mpa = Pixel_Accuracy_Class(confusion)
    miou = Mean_Intersection_over_Union(confusion)
    fwiou = Frequency_Weighted_Intersection_over_Union(confusion)


    return {
            'pa': pa,
            "mpa": mpa,
            'miou': miou,
            'fwiou':fwiou,
            }
