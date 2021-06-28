import os
import cv2
import numpy as np
import torch


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
    if np.max(image) <= 1.5:
        image = image * 255
    path = os.path.join(debug_dir, name + '.jpg')
    assert cv2.imwrite(path, image)
    assert os.path.isfile(path)
    return path


def save_image_stack(image_stack, name, max_count=int('inf'), normalized=False, debug_dir=default_debug_dir):
    # save at most max_count number of images
    save_count = min(max_count, image_stack.shape[0])
    for i in range(save_count):
        save_image(image_stack[i], "{}_{}".format(name, i), normalized, debug_dir)