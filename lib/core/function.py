# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging

import torch
import numpy as np

from lib.core.evaluate import cal_accuracy, cal_roc_auc
from lib.utils.utils import save_checkpoint
from utils.debug_tools import save_image, save_image_stack, clear_debug_image

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion1, criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to train mode
    model.train()

    # freeze / unfreeze hrnet
    if epoch == 0:
        # pass
        logger.info('freezing hrnet')
        model.module.freeze_hrnet()
    elif epoch == 3:
        logger.info('unfreezing hrnet')
        model.module.unfreeze_hrnet()
    # elif epoch >= 5:
    #     logger.info('freezing xray')
    #     model.module.freeze_xray()

    end = time.time()
    debug_input = None
    for i, data in enumerate(train_loader):
        model_input = data['video_frame']
        target_x = data['mask_frame']
        target_c = data['is_fake']
        # if debug_input is None:
        #     debug_input = model_input
        # model_output_x, model_output_c = model(debug_input)
        # print(model_output_x[0][0])
        # input()

        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output_x, output_c = model(model_input)

        target_x = target_x.cuda(non_blocking=True)
        target_c = target_c.cuda(non_blocking=True)

        loss1 = criterion1(output_x, target_x)
        loss2 = criterion2(output_c, target_c)
        loss = loss1 * 100 + loss2

        # compute gradient and do update step
        optimizer.zero_grad()
        # if epoch >= 5:
        #     loss2.backward()
        # else:
        #     loss.backward()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), model_input.size(0))

        # evaluation
        acc = cal_accuracy(output_c, target_c)
        accuracy.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=model_input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, accuracy=accuracy)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('accuracy', accuracy.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        if (i + 1) % config.SAVE_FREQ == 0:
            logger.info('=> saving checkpoint to {}'.format(output_dir))
            save_checkpoint({
                'epoch': epoch,
                'model': config.MODEL.NAME,
                'state_dict': model.module.state_dict(),
                'perf': 0,
                'optimizer': optimizer.state_dict(),
            }, False, output_dir, filename='mini_checkpoint.pth.tar')


def validate(config, val_loader, model, criterion1, criterion2, output_dir, tb_log_dir,
             writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    output_list = np.array([])
    target_list = np.array([])

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            model_input = data['video_frame']
            target_x = data['mask_frame']
            target_c = data['is_fake']

            # compute output
            output_x, output_c = model(model_input)

            # print(torch.amax(model_input.detach().cpu(), dim=(1, 2, 3)), torch.amax(output_x.detach().cpu(), dim=(1, 2, 3)))

            target_x = target_x.cuda(non_blocking=True)
            target_c = target_c.cuda(non_blocking=True)

            loss1 = criterion1(output_x, target_x)
            loss2 = criterion2(output_c, target_c)
            loss = loss1 * 100 + loss2

            if i == 0:
                clear_debug_image()
                save_image_stack(model_input, 'input_{}'.format(i), max_count=3, normalized=True)
                save_image_stack(target_x, 'target_x_{}'.format(i), max_count=3)
                save_image_stack(output_x, 'output_x_{}'.format(i), max_count=3)

            # print('target:')
            # print(target_c.detach().cpu())
            # print('output:')
            # print(output_c.detach().cpu())
            # print('loss 1:')
            # print(loss1 * 100)
            # print('loss 2:')
            # print(loss2)

            # measure accuracy and record loss
            losses.update(loss.item(), model_input.size(0))

            # TODO: Add New Evaluation
            # prec1, prec5 = accuracy(output, target, (1, 5))
            # top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            # evaluation
            acc = cal_accuracy(output_c, target_c)
            output_list = np.append(output_list, output_c.detach().cpu().numpy())
            target_list = np.append(target_list, target_c.detach().cpu().numpy())
            accuracy.update(acc)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # output_list = np.array(output_list).reshape(-1)
        # target_list = np.array(target_list).reshape(-1)
        # print(output_list.shape)
        # print(target_list.shape)
        # print(output_list[np.isnan(output_list)])
        # print(target_list[np.isnan(target_list)])
        auc = cal_roc_auc(output_list, target_list)
        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy {accuracy.avg:.3f}\t' \
              'AUC {auc:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, accuracy=accuracy, auc=auc)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('accuracy', accuracy.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy.avg


def distill(config, train_loader, model_teacher, model_student, criterion1, criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # freeze / unfreeze hrnet
    if epoch == 0:
        model_student.module.freeze_hrnet()
    elif epoch == 3:
        model_student.module.unfreeze_hrnet()

    # switch to train mode
    model_teacher.eval()
    model_student.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        model_input = data['video_frame']
        # target_x = data['mask_frame']
        # target_c = data['is_fake']

        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        target_x, target_c = model_teacher(model_input)
        output_x, output_c = model_student(model_input)

        target_x = target_x.cuda(non_blocking=True)
        target_c = target_c.cuda(non_blocking=True)

        loss1 = criterion1(output_x, target_x.detach())
        loss2 = criterion2(output_c, target_c.detach())
        loss = loss1 * 100 + loss2

        # print('teacher:')
        # print(target_x.detach().cpu())
        # print('student:')
        # print(output_x.detach().cpu())
        # print('loss 1:')
        # print(loss1 * 100)
        # print('loss 2:')
        # print(loss2)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), model_input.size(0))

        # evaluation
        acc = cal_accuracy(output_c, target_c)
        accuracy.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=model_input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, accuracy=accuracy)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('accuracy', accuracy.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        if (i + 1) % config.SAVE_FREQ == 0:
            logger.info('=> saving checkpoint to {}'.format(output_dir))
            save_checkpoint({
                'epoch': epoch,
                'model': config.MODEL.NAME,
                'state_dict': model_student.module.state_dict(),
                'perf': 0,
                'optimizer': optimizer.state_dict(),
            }, False, output_dir, filename='mini_checkpoint.pth.tar')


def smooth_distill(config, train_loader, model_teacher, model_student, criterion1, criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # freeze / unfreeze hrnet
    if epoch == 0:
        model_student.module.freeze_hrnet()
    elif epoch == 3:
        model_student.module.unfreeze_hrnet()

    # switch to train mode
    model_teacher.eval()
    model_student.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        model_input_teacher = data['video_frame']
        model_input_teacher.requires_grad = True
        model_input_student = model_input_teacher.detach().clone()
        target_x = data['mask_frame']
        target_c = data['is_fake']

        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        teacher_x, teacher_c = model_teacher(model_input_teacher)

        torch.sum(teacher_c).backward()
        # print(model_input_teacher.grad.data)
        model_input_neighbour = get_input_neighbour(model_input_teacher, model_input_teacher.grad.data)
        neighbour_x, neighbour_c = model_teacher(model_input_neighbour)
        clear_debug_image()
        save_image_stack(model_input_teacher, 'teacher input', 5, normalized=True)
        save_image_stack(model_input_neighbour, 'neighbour input', 5, normalized=True)
        save_image_stack(teacher_x, 'teacher output', 5)
        save_image_stack(neighbour_x, 'neighbour output', 5)

        output_x, output_c = model_student(model_input_teacher)

        teacher_x = teacher_x.cuda(non_blocking=True)
        teacher_c = teacher_c.cuda(non_blocking=True)

        loss1 = criterion1(output_x, teacher_x.detach())
        loss2 = criterion2(output_c, teacher_c.detach())
        loss = loss1 * 100 + loss2

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), model_input_teacher.size(0))

        # evaluation
        acc = cal_accuracy(output_c, teacher_c)
        accuracy.update(acc)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=model_input_teacher.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, accuracy=accuracy)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('accuracy', accuracy.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        if (i + 1) % config.SAVE_FREQ == 0:
            logger.info('=> saving checkpoint to {}'.format(output_dir))
            save_checkpoint({
                'epoch': epoch,
                'model': config.MODEL.NAME,
                'state_dict': model_student.module.state_dict(),
                'perf': 0,
                'optimizer': optimizer.state_dict(),
            }, False, output_dir, filename='mini_checkpoint.pth.tar')


def get_input_neighbour(input_data, grad):
    target_mean = 0.0005
    print(torch.mean(grad, dim=1))
    displacement = grad * target_mean / torch.mean(grad, dim=1, keepdim=True)
    input_neighbour = input_data - displacement
    return input_neighbour


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
