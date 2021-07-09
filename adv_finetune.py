# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import lib.models.cls_hrnet as cls_hrnet
from lib.models.xray_net import XRayNet
import utils.image_transforms as custom_transforms
from utils.model_loader import Args, construct_model
from utils.xray_dataset import XRayDataset

from lib.config import config
from lib.config import update_config
from lib.core.function import validate, train, adv_finetune
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import get_optimizer, save_checkpoint, create_logger

cfg_path = 'experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100_adapted_linux_adv.yaml'
pretrained_model = 'hrnetv2_w18_imagenet_pretrained.pth'
base_model_path = "./output/ILSVRC/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100_adapted_linux_exp05/model_best.pth.tar"


def main():
    model, args, config = construct_model(cfg_path, pretrained_model)
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    # added hrnet model pretrained loading
    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        # model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        model.load_hrnet_pretrained(torch.load(config.TEST.MODEL_FILE))

    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, 'lib/models'), models_dst_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    # criterion = torch.nn.CrossEntropyLoss().cuda()
    criterion1 = torch.nn.BCELoss().cuda()
    criterion2 = torch.nn.BCELoss().cuda()

    optimizer = get_optimizer(config, model)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        else:  # load initial model trained before
            assert os.path.isfile(base_model_path)
            model_state = torch.load(base_model_path)
            last_epoch = 0
            print(model_state.keys())
            if 'state_dict' in model_state:
                model.load_state_dict(model_state['state_dict'])
            else:
                model.load_state_dict(model_state)
            logger.info("=> loaded base model: {}".format(base_model_path))

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
            last_epoch-1
        )

    # Data loading code
    # traindir = os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET)
    # valdir = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_SET)
    #
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    train_dataset = XRayDataset(
        './data/train_image_selected.csv',
        transforms.Compose([
            custom_transforms.ImageToOne(),
            custom_transforms.MaskToXray(),
            custom_transforms.ToTensor(cuda=False),
            custom_transforms.ColorJitter(),
            custom_transforms.Rescale(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            custom_transforms.RandomCrop(config.MODEL.IMAGE_SIZE[0]),
            # custom_transforms.PiecewiseAffine(),
            # custom_transforms.Affine(),
            # custom_transforms.LinearContrast(),
            # custom_transforms.HueAndSaturation(),
            # custom_transforms.Grayscale(enabled=config.GRAYSCALE),
            custom_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])
    )

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # valid_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
    #         transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True
    # )

    valid_dataset = XRayDataset(
        './data/val_image_selected.csv',
         transforms.Compose([
             # TODO: Change Random Crop to Centre Crop
             custom_transforms.ImageToOne(),
             custom_transforms.MaskToXray(),
             custom_transforms.ToTensor(cuda=False),
             custom_transforms.Rescale(int(config.MODEL.IMAGE_SIZE[0])),
             # custom_transforms.Grayscale(enabled=config.GRAYSCALE),
             custom_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
         ]))

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # train for one epoch
        adv_finetune(config, train_loader, model, criterion1, criterion2, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        lr_scheduler.step()
        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion1, criterion2,
                                  final_output_dir, tb_log_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    logger.info('best validation accuracy: {}'.format(best_perf))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
