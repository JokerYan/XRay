import os
import re
import cv2
import random
import numpy as np
from glob import glob

import torch
from tqdm import tqdm
import torchvision.transforms as transforms

from lib.config import config
from utils.data_config import fake_root_dir, video_folder, real_root_dir, mask_folder, \
    fake_video_dir, fake_mask_dir
import utils.image_transforms as custom_transforms


def load_video_paths(video_dir, mask_dir):
    video_mask_list = []
    valid_video_count = 0
    total_video_count = 0
    for video_path in glob(video_dir + "*.mp4", recursive=True):
        assert video_path.endswith("mp4")
        video_path = video_path.replace("\\", "/")
        if mask_dir is not None:
            dataset_name = re.search(fake_root_dir + r"([^/\\]*)[/\\]" + video_folder, video_path).group(1)
        else:
            dataset_name = re.search(real_root_dir + r"([^/\\]*)[/\\](?:[^/\\]*)[/\\]", video_path).group(1)
        video_name = re.search(r'[/\\]([\d\w_]+.mp4)', video_path).group(1)
        if mask_dir is None:  # real video
            mask_path = None
        else:
            mask_path = os.path.join(fake_root_dir, dataset_name, mask_folder, video_name)
            if not os.path.isfile(mask_path):
                continue
        total_video_count += 1
        if mask_path is None or get_video_frame_count(video_path) == get_video_frame_count(mask_path):
            valid_video_count += 1
            video_mask_list.append((video_path, mask_path))
    print('valid video loaded: {}/{}'.format(valid_video_count, total_video_count))
    return video_mask_list


def get_video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    return frame_count


def get_random_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    selected_frame_index = random.randint(0, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame_index)
    ret, frame = cap.read()

    return selected_frame_index, frame


def get_frame_from_video(video_path, frame_idx, gray_scale=False):
    cap = cv2.VideoCapture(video_path)
    if not frame_idx < get_video_frame_count(video_path):
        print(frame_idx, get_video_frame_count(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if gray_scale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame


def get_random_frame_from_list(video_mask_list):
    video_idx = random.randint(0, len(video_mask_list) - 1)
    frame_idx, video_frame = get_random_frame_from_video(video_mask_list[video_idx][0])
    mask_frame = get_frame_from_video(video_mask_list[video_idx][1], frame_idx)
    return video_frame, mask_frame


def get_blank_mask_from_size(size):
    mask_frame = np.zeros(size, dtype=np.float)
    return mask_frame


def get_impulse_from_size(size, x, y, c, step=1, value=255, reference=0):
    # numpy H x W x C
    assert x < size[0]
    assert y < size[1]
    assert c < size[2]
    impulse = np.ones(size, dtype=np.float) * reference
    for i in range(step):
        for j in range(step):
            if x + i < size[0] and y + j < size[1]:
                # # TODO: all channel test
                # for c in range(size[2]):
                impulse[x + i][y + j][c] = value
    return impulse


def show_image(image, title=""):
    target_width = 400
    target_height = int(target_width * image.shape[0] / image.shape[1])
    resized_image = cv2.resize(image, (target_width, target_height))

    cv2.imshow(title, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_normalized_images(video_frame, mask_frame, title):
    video_frame = video_frame.cpu().clone().detach().numpy()
    mask_frame = mask_frame.cpu().clone().detach().numpy()
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    video_frame = video_frame.transpose((1, 2, 0))
    mask_frame = mask_frame.transpose((1, 2, 0))
    video_frame = np.multiply(video_frame, std)
    mean = np.multiply(np.ones_like(video_frame), mean)
    video_frame = video_frame + mean
    # video_frame = video_frame * 255
    # mask_frame = mask_frame * 255
    mask_frame = np.tile(mask_frame, [3])
    show_image(np.vstack((video_frame, mask_frame)), title)


def save_image_to_disk(image, dir, filename):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, filename)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        image = image.transpose((1, 2, 0))
        image = image * 255
    cv2.imwrite(path, image)
    assert os.path.isfile(path)
    return path


def generate_impulse_image_and_csv(image_dir, csv_path):
    size = 256
    channel = 3
    step = 1

    impulse_cap = 256
    impulse_interval = 64
    impulse_reference = 128

    path_list = []
    for x in range(128, 132, step):
        for y in range(128, 132, step):
    # for x in range(0, size, step):
    #     for y in range(0, size, step):
            for c in range(channel):
                for v in range(0, impulse_cap, impulse_interval):
                    impulse_value = v + impulse_interval
                    filename = r'x{}_y{}_c{}_v{}.jpg'.format(x, y, c, impulse_value)
                    img_size = [size, size, channel]  # numpy H x W x C
                    impulse_image = get_impulse_from_size(img_size, x, y, c, step, impulse_value, impulse_reference)
                    path = save_image_to_disk(impulse_image, image_dir, filename)
                    path_list.append(path)

    csv_lines = [path + ',,0' for path in path_list]  # img_path, no mask, not fake
    with open(csv_path, 'w+') as csv_file:
        csv_file.write('\n'.join(csv_lines))


def main():
    # video_mask_list = load_from_csv('./data/train_image_selected.csv')
    video_mask_list = []
    with open('./data/train_image_selected.csv', "r") as csv_file:
        for line in csv_file:
            line_parts = line.strip().split(",")
            video_path = line_parts[0]
            mask_path = line_parts[1] if line_parts[1] else None
            is_fake = np.float32(line_parts[2])
            video_mask_list.append([video_path, mask_path, is_fake])
    # video_mask_list = load_video_paths(fake_video_dir, fake_mask_dir)
    # video_frame, mask_frame = get_random_frame_from_list(video_mask_list)
    video_path, mask_path, _ = video_mask_list[random.randint(0, len(video_mask_list) - 1)]
    video_frame = cv2.imread(video_path)
    if mask_path:
        mask_frame = cv2.imread(mask_path)
    else:
        mask_frame = get_blank_mask_from_size(video_frame.shape)
    sample = {'video_frame': video_frame, 'mask_frame': mask_frame, 'is_fake': 1}
    transform = transforms.Compose([
        custom_transforms.Rescale(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
        custom_transforms.RandomCrop(config.MODEL.IMAGE_SIZE[0]),
        custom_transforms.PiecewiseAffine(),
        custom_transforms.Affine(),
        custom_transforms.LinearContrast(),
        custom_transforms.HueAndSaturation(),
        # custom_transforms.ImageToOne(),
        custom_transforms.MaskToXray(),
    ])
    sample = transform(sample)
    show_image(np.vstack((sample['video_frame'], sample['mask_frame'])))
    # show_image(sample['video_frame'])

    # count_list = []
    # for video, mask in tqdm(video_mask_list):
    #     count_list.append(get_video_frame_count(video))
    # print(count_list)


if __name__ == "__main__":
    main()