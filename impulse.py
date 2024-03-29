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
import sys
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# import _init_paths
# import models
# from config import config
# from config import update_config
# from core.function import validate
# from utils.modelsummary import get_model_summary
# from utils.utils import create_logger

import lib.models.cls_hrnet as cls_hrnet
from lib.models.xray_net import XRayNet
import utils.image_transforms as custom_transforms
from utils.model_loader import Args, construct_model
from utils.image_preprocessing import generate_impulse_image_and_csv
from utils.xray_dataset import XRayDataset

from lib.config import config
from lib.config import update_config
from lib.core.function import validate, train
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import get_optimizer, save_checkpoint, create_logger


cfg_path = 'experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100_adapted_linux.yaml'
pretrained_model = 'hrnetv2_w64_imagenet_pretrained.pth'


def main():
    model, args, config = construct_model(cfg_path, pretrained_model)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model, dump_input))

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        # model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
        model.load_hrnet_pretrained(torch.load(config.TEST.MODEL_FILE))
    else:
        # model_state_file = os.path.join(final_output_dir,
        #                                 'final_state.pth.tar')
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model_state = torch.load(model_state_file)
        if 'state_dict' in model_state:
            model.load_state_dict(model_state['state_dict'])
        else:
            model.load_state_dict(model_state)

    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion1 = torch.nn.BCELoss().cuda()
    criterion2 = torch.nn.BCELoss().cuda()

    # Data loading code
    impulse_image_dir = os.path.expanduser('~/zhiwen/XRay/Data/ImpulseImages/')
    impulse_csv_path = './data/impulse.csv'
    if not os.path.isfile(impulse_csv_path):
        generate_impulse_image_and_csv(impulse_image_dir, impulse_csv_path)
    valid_dataset = XRayDataset(
        impulse_csv_path,
        transforms.Compose([
            # TODO: Change Random Crop to Centre Crop
            custom_transforms.Rescale(int(config.MODEL.IMAGE_SIZE[0])),
            custom_transforms.StrengthenImpulse(multiplier=100),
            custom_transforms.ImageToOne(),
            custom_transforms.MaskToXray(),
            custom_transforms.ToTensor(),
            custom_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        ]))

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(config, valid_loader, model, criterion1, criterion2, final_output_dir,
             tb_log_dir, None, show_image=False, save_image=True)


if __name__ == '__main__':
    main()
