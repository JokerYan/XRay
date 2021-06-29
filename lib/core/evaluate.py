# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from sklearn.metrics import roc_auc_score


# def cal_accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

def cal_accuracy(output, target):
    with torch.no_grad():
        ones = torch.ones_like(output)
        zeros = torch.zeros_like(output)
        prediction = torch.where(output > 0.5, ones, zeros)
        target = torch.where(target > 0.5, ones, zeros)
        # print(prediction, target)
        return 1 - torch.mean(torch.abs(prediction - target))


def cal_roc_auc(output, target):
    with torch.no_grad():
        if torch.is_tensor(target):
            target = target.detach().cpu()
        if torch.is_tensor(output):
            output = output.detach().cpu()
        return roc_auc_score(target, output)