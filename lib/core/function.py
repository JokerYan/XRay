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

from lib.core.evaluate import cal_accuracy
from lib.utils.utils import save_checkpoint

from utils.image_preprocessing import show_normalized_images, save_image_to_disk


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion1, criterion2, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # freeze / unfreeze hrnet
    if epoch == 0:
        model.module.freeze_hrnet()
    elif epoch == 3:
        model.module.unfreeze_hrnet()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        input = data['video_frame']
        target_x = data['mask_frame']
        target_c = data['is_fake']

        # measure data loading time
        data_time.update(time.time() - end)
        #target = target - 1 # Specific for imagenet

        # compute output
        output_x, output_c = model(input)

        target_x = target_x.cuda(non_blocking=True)
        target_c = target_c.cuda(non_blocking=True)

        loss1 = criterion1(output_x, target_x)
        loss2 = criterion2(output_c, target_c)
        loss = loss1 * 100 + loss2

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

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
                      speed=input.size(0)/batch_time.val,
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
             writer_dict=None, show_image=False, save_image=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            input = data['video_frame']
            target_x = data['mask_frame']
            target_c = data['is_fake']

            # compute output
            output_x, output_c = model(input)

            if show_image:
                title = str(target_c[0].cpu().clone().detach().numpy()) + "   " + \
                        str(output_c[0].cpu().clone().detach().numpy())
                show_normalized_images(input[0], output_x[0], title)
            if save_image:
                output_image_dir = os.path.join(output_dir, 'output_images')
                for k in range(input.size()[0]):
                    output_image = output_x[k]
                    output_idx = i * input.size()[0] + k
                    output_image_name = str(output_idx) + '.jpg'
                    save_image_to_disk(output_image, output_image_dir, output_image_name)

            target_x = target_x.cuda(non_blocking=True)
            target_c = target_c.cuda(non_blocking=True)

            loss1 = criterion1(output_x, target_x)
            loss2 = criterion2(output_c, target_c)
            loss = loss1 * 100 + loss2

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # TODO: Add New Evaluation
            # prec1, prec5 = accuracy(output, target, (1, 5))
            # top1.update(prec1[0], input.size(0))
            # top5.update(prec5[0], input.size(0))

            # evaluation
            acc = cal_accuracy(output_c, target_c)
            accuracy.update(acc)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy {accuracy.avg:.3f}\t'.format(
                  batch_time=batch_time, loss=losses, accuracy=accuracy)
        logger.info(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('accuracy', accuracy.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy.avg


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
