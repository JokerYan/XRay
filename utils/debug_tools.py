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
    model = model.cuda()

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
    valid_dataset_transformed = XRayDataset(
        './data/val_image_selected.csv',
         transforms.Compose([
             # TODO: Change Random Crop to Centre Crop
             custom_transforms.ImageToOne(),
             custom_transforms.MaskToXray(),
             custom_transforms.ToTensor(cuda=False),
             custom_transforms.ColorJitter(),
             custom_transforms.Rescale(int(config.MODEL.IMAGE_SIZE[0])),
             # custom_transforms.Grayscale(enabled=config.GRAYSCALE),
             custom_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
         ]))

    data = valid_dataset[0]
    model_input = torch.unsqueeze(data['video_frame'], 0)
    target_x = torch.unsqueeze(data['mask_frame'], 0)
    # target_c = torch.unsqueeze(data['is_fake'], 0)
    output_x, output_c = model(model_input)

    data_transformed = valid_dataset_transformed[0]
    model_input_transformed = torch.unsqueeze(data_transformed['video_frame'], 0)
    target_x_transformed = torch.unsqueeze(data['mask_frame'], 0)
    # target_c_transformed = torch.unsqueeze(data['is_fake'], 0)
    output_x_transformed, output_c_transformed = model(model_input_transformed)

    clear_debug_image()
    save_image_stack(model_input, 'model_input', normalized=True)
    save_image_stack(target_x, 'target')
    save_image_stack(output_x, 'output')
    save_image_stack(model_input_transformed, 'model_input_transformed', normalized=True)
    save_image_stack(target_x_transformed, 'target_transformed')
    save_image_stack(output_x_transformed, 'output_transformed')

