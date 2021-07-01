import os
import math
import cv2
import torch
import glob
import numpy as np
from torch.backends import cudnn
import torchvision.transforms as transforms

from lib import config
from lib.config import update_config
from lib.models.xray_net import XRayNet
from utils.model_loader import Args, construct_model
from utils.xray_dataset import XRayDataset
import utils.image_transforms as custom_transforms

default_debug_dir = os.path.join('.', 'debug')


def save_image(image, name, normalized=False, debug_dir=default_debug_dir):
    if not os.path.isdir(debug_dir):
        os.makedirs(debug_dir, exist_ok=True)
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
        image = image.transpose((1, 2, 0))
    if normalized:
        mean = np.asarray([0.485, 0.456, 0.406])
        std = np.asarray([0.229, 0.224, 0.225])
        image = np.multiply(image, std)
        mean = np.multiply(np.ones_like(image), mean)
        image = image + mean
    if np.max(image) <= 10:
        image = image * 255
    path = os.path.join(debug_dir, name + '.jpg')
    assert cv2.imwrite(path, image)
    assert os.path.isfile(path)
    return path


def save_image_stack(image_stack, name, max_count=math.inf, normalized=False, debug_dir=default_debug_dir):
    # save at most max_count number of images
    save_count = int(min(max_count, image_stack.shape[0]))
    for i in range(save_count):
        save_image(image_stack[i], "{}_{}".format(name, i), normalized, debug_dir)


def clear_debug_image(debug_dir=default_debug_dir):
    for f in glob.glob(os.path.join(debug_dir, '*.jpg')):
        os.remove(f)
    print('debug image cleared')


def visualize_transform():
    cfg_path = 'experiments/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100_adapted_linux.yaml'
    pretrained_model = 'hrnetv2_w64_imagenet_pretrained.pth'
    model, args, config = construct_model(cfg_path, pretrained_model)
    model_state = torch.load(os.path.join(
        'output/ILSVRC/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100_adapted_linux',
        'model_best.pth.tar'))
    model.load_state_dict(model_state)
    gpus = list(config.GPUS)
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

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
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    for i, data in enumerate(valid_loader):
        if i > 0:
            break
        model_input = data['video_frame']
        target_x = data['mask_frame']
        target_c = data['is_fake']
        output_x, output_c = model(model_input)

