import os
import re
import cv2
import random
import numpy as np
from glob import glob
from tqdm import tqdm

from utils.image_transforms import ImageToOne, MaskToXray

default_root_dir = "C:/Code/Projects/XRay/Data/FaceForensic/manipulated_sequences/DeepFakeDetection/"
default_video_dir = os.path.join(default_root_dir, "c23/videos/")
default_mask_dir = os.path.join(default_root_dir, "masks/videos/")

default_real_root_dir = r"C:/Code/Projects/XRay/Data/FaceForensic/original_sequences/actors/"
default_real_video_dir = os.path.join(default_real_root_dir, "c23/videos/")

# set mask dir to None if it is a real video
def load_video_paths(video_dir, mask_dir):
    video_mask_list = []
    valid_video_count = 0
    total_video_count = 0
    for video_path in glob(video_dir + "*"):
        assert video_path.endswith("mp4")
        video_name = re.search(r'[/\\]([\d\w_]+.mp4)', video_path).group(1)
        if mask_dir is None:  # real video
            mask_path = None
        else:
            mask_path = os.path.join(mask_dir, video_name)
            assert os.path.isfile(mask_path)
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
    mask_frame = np.empty(size, dtype=np.float)
    return mask_frame


def show_image(image):
    target_width = 800
    target_height = int(target_width * image.shape[0] / image.shape[1])
    resized_image = cv2.resize(image, (target_width, target_height))

    cv2.imshow("", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    video_mask_list = load_video_paths(default_video_dir, default_mask_dir)
    video_frame, mask_frame = get_random_frame_from_list(video_mask_list)
    sample = {'video_frame': video_frame, 'mask_frame': mask_frame}
    to_one_transformer = ImageToOne()
    to_xray_transformer = MaskToXray()
    sample = to_one_transformer(sample)
    sample = to_xray_transformer(sample)
    show_image(np.vstack((sample['video_frame'], sample['mask_frame'])))

    # count_list = []
    # for video, mask in tqdm(video_mask_list):
    #     count_list.append(get_video_frame_count(video))
    # print(count_list)

if __name__ == "__main__":
    main()