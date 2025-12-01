from __future__ import division

import numpy as np
import six
import torch
import torch.nn as nn


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.

    """
    pred_labels = iter(pred_labels)  # (352, 480)
    gt_labels = iter(gt_labels)  # (352, 480)

    n_class = 1
    confusion = np.zeros((n_class, n_class), dtype=np.int64)  # (12, 12)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        # if pred_label.ndim != 2 or gt_label.ndim != 2:
        #     raise ValueError('ndim of labels should be two.')
        # if pred_label.shape != gt_label.shape:
        #     raise ValueError('Shape of ground truth and prediction should'
        #                      ' be same.')
        pred_label = pred_label.flatten()  # (168960, )
        gt_label = gt_label.flatten()  # (168960, )

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2) \
            .reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.

    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.

    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.

    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou[:-1]
    # return iou


def eval_semantic_segmentation(pred_labels, gt_labels, preout, gtout):
    """Evaluate metrics used in Semantic Segmentation.

    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.

    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`

    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
            For example, this is a list of labels
            :obj:`[label_0, label_1, ...]`, where
            :obj:`label_i.shape = (H_i, W_i)`.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.

    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    """

    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)
    JS = get_JS(preout, gtout)
    DC = get_dice(preout, gtout)
    RVD = rvd(preout, gtout)
    VOE = _VOE(preout, gtout)
    SP = _specificity(preout, gtout)
    SE = _sensitivity(preout, gtout)
    PC = _precision(preout, gtout)
    RE = _recall(preout, gtout)
    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy[:-1]),
            'JS': JS,
            'DC': DC,
            'SP': SP,
            'SE': SE,
            'PC': PC,
            'RE': RE,
            'RVD': RVD,
            'VOE': VOE
            }
    # 'mean_class_accuracy': np.nanmean(class_accuracy)}


# JC/VOE
def get_JS(preout, gtout):
    # JS : Jaccard similarity

    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    intersection = torch.sum((preout + gtout) == 0)
    union = torch.sum((preout + gtout) == 0) + torch.sum((preout + gtout) == 1)

    JS = float(intersection) / (float(union) + 1e-6)

    return JS


def get_dice(preout, gtout):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
    preout = torch.Tensor(preout)

    gtout = torch.Tensor(gtout)

    intersection = torch.sum((preout + gtout) == 0)

    union = torch.sum((preout + gtout) == 0) + torch.sum((preout + gtout) == 0) + torch.sum((preout + gtout) == 1)

    return float(2 * intersection) / float(union + 1e-6)


def rvd(preout, gtout):

    preout = torch.Tensor(preout)

    gtout = torch.Tensor(gtout)


    a = torch.sum(preout == 0)

    b = torch.sum(gtout == 0)


    return float(a - b) / float(b + 1e-6)


def _VOE(preout, gtout):
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    a = torch.sum(preout == 0)
    b = torch.sum(gtout == 0)

    VOE = 2 * (a - b) / (a + b)
    return VOE


def _specificity(preout, gtout):

    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TN = ((preout == 1) & (gtout == 1))
    FP = ((preout == 0) & (gtout == 1))

    return float(torch.sum(TN)) / float(torch.sum(TN + FP) + 1e-6)


def _sensitivity(preout, gtout):

    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FN = ((preout == 1) & (gtout == 0))

    return float(torch.sum(TP)) / float(torch.sum(TP + FN) + 1e-6)


def _precision(preout, gtout):

    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FP = ((preout == 0) & (gtout == 1))

    return float(torch.sum(TP)) / float(torch.sum(TP + FP) + 1e-6)


def _recall(preout, gtout):

    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FN = ((preout == 1) & (gtout == 0))

    return float(torch.sum(TP)) / float(torch.sum(TP + FN) + 1e-6)


if __name__ == "__main__":
    import torch as t
    import numpy

    print('-----' * 5)
    pred_labels = numpy.random.randint(6, 256, 256)
    gt_labels = numpy.random.randint(6, 256, 256)
    preout = numpy.random.randint(6, 256, 256)
    gtout = numpy.random.randint(6, 256, 256)
# import numpy as np
# import torch

# def eval_semantic_segmentation(pred_masks, true_masks):
#     """
#     为二分类分割任务计算一套标准的评估指标。
#     假设 类别0=背景, 类别1=前景。

#     Args:
#         pred_masks (np.ndarray): 预测的二值化mask，形状 [N, H, W]，值为0或1。
#         true_masks (np.ndarray): 真实的二值化mask，形状 [N, H, W]，值为0或1。

#     Returns:
#         dict: 包含各种平均指标的字典。
#     """
#     if isinstance(pred_masks, list):
#         pred_masks = np.array(pred_masks)
#     if isinstance(true_masks, list):
#         true_masks = np.array(true_masks)

#     pred_flat = pred_masks.flatten()
#     true_flat = true_masks.flatten()
    
#     TP = np.sum((pred_flat == 1) & (true_flat == 1))
#     TN = np.sum((pred_flat == 0) & (true_flat == 0))
#     FP = np.sum((pred_flat == 1) & (true_flat == 0))
#     FN = np.sum((pred_flat == 0) & (true_flat == 1))
    
#     epsilon = 1e-6

#     # --- 计算指标 ---
    
#     iou = (TP + epsilon) / (TP + FP + FN + epsilon)
#     dice = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
    
#     # Sensitivity (前景准确率, Class 1 Accuracy)
#     sensitivity = (TP + epsilon) / (TP + FN + epsilon)
    
#     # Specificity (背景准确率, Class 0 Accuracy)
#     specificity = (TN + epsilon) / (TN + FP + epsilon)
    
#     precision = (TP + epsilon) / (TP + FP + epsilon)
    
#     # --- START OF MODIFICATION ---
#     # 组合成 class_accuracy 数组
#     # 索引0: 背景准确率
#     # 索引1: 前景准确率
#     class_accuracy = np.array([specificity, sensitivity])
#     # --- END OF MODIFICATION ---

#     # Mean Class Accuracy
#     mean_class_accuracy = (sensitivity + specificity) / 2
    
#     pixel_accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)

#     # 封装成字典返回
#     return {
#         'mIoU': iou,
#         'DICE': dice,
#         'Jaccard': iou,
#         'mAcc': mean_class_accuracy,
#         'pixel_accuracy': pixel_accuracy,
#         'sensitivity': sensitivity,
#         'specificity': specificity,
#         'precision': precision,
#         'class_accuracy': class_accuracy, # <-- 新增的键
#     }