import cv2
import torch
import time
import numpy as np
from skimage import transform as skimage_transforms
import torchvision.transforms as torch_transforms

import imgaug as ia
import imgaug.augmenters as iaa


class ImageToOne:
    def __call__(self, sample):
        return {
            'video_frame': sample['video_frame'] / 255.0,
            'mask_frame': sample['mask_frame'] / 255.0,
            'is_fake': sample['is_fake']
        }


class MaskToXray:
    def threshold(self, src):
        dst = np.where(src > 0.2, 1.0, 0.0)
        return dst

    def blur(self, src, kernal_size=3):
        kernal = np.ones((kernal_size, kernal_size), np.float32) / (kernal_size ** 2)
        dst = cv2.filter2D(src, -1, kernal)
        return dst

    def __call__(self, sample):
        thresh = self.threshold(sample['mask_frame'])
        blurred = self.blur(thresh)
        xray = 4 * np.multiply(blurred, np.ones_like(blurred) - blurred)
        return {'video_frame': sample['video_frame'], 'mask_frame': xray, 'is_fake': sample['is_fake']}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video_frame, mask_frame = sample['video_frame'], sample['mask_frame']

        h, w = video_frame.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        video_frame = skimage_transforms.resize(video_frame, (new_h, new_w), anti_aliasing=True)
        mask_frame = skimage_transforms.resize(mask_frame, (new_h, new_w), anti_aliasing=True)

        return {'video_frame': video_frame, 'mask_frame': mask_frame, 'is_fake': sample['is_fake']}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video_frame, mask_frame = sample['video_frame'], sample['mask_frame']

        h, w = video_frame.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        video_frame = video_frame[top: top + new_h, left: left + new_w]
        mask_frame = mask_frame[top: top + new_h, left: left + new_w]

        return {'video_frame': video_frame, 'mask_frame': mask_frame, 'is_fake': sample['is_fake']}


class PiecewiseAffine(object):
    def __init__(self):
        self.ia_piece_affine = iaa.PiecewiseAffine(scale=(0, 0.02))

    def __call__(self, sample):
        start_time = time.time()
        ia_piece_affine_det = self.ia_piece_affine.to_deterministic()
        video_frame = ia_piece_affine_det(image=sample['video_frame'])
        mask_frame = ia_piece_affine_det(image=sample['mask_frame'])
        print('Piecewise Affine', time.time() - start_time)
        return {'video_frame': video_frame, 'mask_frame': mask_frame, 'is_fake': sample['is_fake']}


class Affine(object):
    def __init__(self):
        self.ia_affine = iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-5, 5),
            shear=(-5, 5)
        )

    def __call__(self, sample):
        start_time = time.time()
        ia_affine_det = self.ia_affine.to_deterministic()
        video_frame = ia_affine_det(image=sample['video_frame'])
        mask_frame = ia_affine_det(image=sample['mask_frame'])
        print('Affine', time.time() - start_time)
        return {'video_frame': video_frame, 'mask_frame': mask_frame, 'is_fake': sample['is_fake']}


class LinearContrast(object):
    def __init__(self):
        self.linear_contrast = iaa.LinearContrast((0.9, 1.1))

    def __call__(self, sample):
        start_time = time.time()
        linear_contrast_det = self.linear_contrast.to_deterministic()
        video_frame = linear_contrast_det(image=sample['video_frame'])
        mask_frame = sample['mask_frame']
        print('Contrast', time.time() - start_time)
        return {'video_frame': video_frame, 'mask_frame': mask_frame, 'is_fake': sample['is_fake']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        video_frame, mask_frame = sample['video_frame'], sample['mask_frame']

        # swap color axis because
        # numpy video_frame: H x W x C
        # torch video_frame: C X H X W
        video_frame = video_frame.transpose((2, 0, 1))
        # mask_frame = mask_frame.transpose((2, 0, 1))
        mask_frame = np.expand_dims(mask_frame, axis=0)
        return {'video_frame': torch.from_numpy(video_frame).float(),
                'mask_frame': torch.from_numpy(mask_frame).float(),
                'is_fake': sample['is_fake']}


class Normalize(object):
    def __init__(self, mean, std):
        self.image_normalize = torch_transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        return {
            'video_frame': self.image_normalize(sample['video_frame']),
            'mask_frame': sample['mask_frame'],
            'is_fake': sample['is_fake']
        }