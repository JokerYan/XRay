import os
import re
import cv2
import time
from skimage import io as skiio
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from utils.data_preperation import load_from_csv, write_to_csv, concate_csv

output_dir = r"C:/Code/Projects/XRay/Data/FaceForensicImage/"
train_input_csv = "./data/train_video.csv"
train_output_csv = "./data/train_image.csv"
val_input_csv = "./data/val_video.csv"
val_output_csv = "./data/val_image.csv"
target_size = int(256 / 0.875)

# output_dir = r"C:/Code/Projects/XRay/Data/FaceForensicImage/"
# train_input_csv = "./data/train_video_youtube.csv"
# train_output_csv = "./data/train_image_youtube.csv"
# val_input_csv = "./data/val_video_youtube.csv"
# val_output_csv = "./data/val_image_youtube.csv"
# target_size = int(256 / 0.875)


def save_all_frames_from_csv(input_csv, output_csv):
    video_paths = load_from_csv(input_csv)
    with Pool(cpu_count()) as pool:
        output_path_list = list(tqdm(pool.imap_unordered(save_video_pair, video_paths), total=len(video_paths)))
    # concat all
    output_paths = []
    for video_frame_paths in output_path_list:
        output_paths += video_frame_paths
    write_to_csv(output_csv, output_paths)


def save_video_pair(video_path_pair):
    video_path, mask_path, is_fake = video_path_pair
    output_paths = []
    video_image_paths = save_video_frames(video_path, "video")
    if mask_path is None:
        mask_image_paths = [None for _ in video_image_paths]
    else:
        mask_image_paths = save_video_frames(mask_path, "mask")

    assert len(video_image_paths) == len(mask_image_paths)
    for i in range(len(video_image_paths)):
        output_paths.append((video_image_paths[i], mask_image_paths[i]))
    return output_paths


def save_video_frames(src_path, suffix, overwrite=False):
    src_path = src_path.replace("\\", "/")
    dataset = re.search(r'[/\\]([^/\\]*)/[^/\\]*/[^/\\]*[/\\]([^/\\]*)\.mp4', src_path).group(1)
    video_name = re.search(r'[/\\]([^/\\]*)/[^/\\]*/[^/\\]*[/\\]([^/\\]*)\.mp4', src_path).group(2)
    # print(os.path.isfile(src_path), dataset, video_name)
    assert os.path.isfile(src_path), src_path
    cap = cv2.VideoCapture(src_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_paths = []
    for i in range(frame_count):
        output_name = dataset + "_" + video_name + "_" + suffix + "_" + str(i) + ".jpg"
        output_video_dir = os.path.join(output_dir, dataset + "_" + video_name)
        if not os.path.isdir(output_video_dir):
            os.makedirs(output_video_dir, exist_ok=True)
        output_path = os.path.join(output_video_dir, output_name)
        if not overwrite and os.path.isfile(output_path):
            save_paths.append(output_path)
            continue

        ret, frame = cap.read()
        if frame is None:
            print(src_path, i)
            continue
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        if overwrite or not os.path.isfile(output_path):
            cv2.imwrite(output_path, frame)
        save_paths.append(output_path)
    return save_paths


def detect_premature_image(image_path):
    with open(image_path, 'rb') as f:
        check_chars = f.read()[-2:]
    if check_chars != b'\xff\xd9':
        print('Not complete image')
        return False
    return True


def main():
    os.makedirs(output_dir, exist_ok=True)
    save_all_frames_from_csv(train_input_csv, train_output_csv)
    save_all_frames_from_csv(val_input_csv, val_output_csv)

    # # walk through all frames to ensure complete jpg saved
    # train_video_frame_list = load_from_csv(train_output_csv)
    # val_video_frame_list = load_from_csv(val_output_csv)
    # # print("training set")
    # # for i in tqdm(range(len(train_video_frame_list))):
    # #     video_path, mask_path, is_fake = train_video_frame_list[i]
    # #     if not detect_premature_image(video_path):
    # #         print(i, video_path)
    # #     if mask_path is None and not detect_premature_image(mask_path):
    # #         print(i, mask_path)
    # print("validation set")
    # for i in tqdm(range(len(val_video_frame_list))):
    #     video_path, mask_path, is_fake = train_video_frame_list[i]
    #     if not detect_premature_image(video_path):
    #         print(i, video_path)
    #     if mask_path is not None and not detect_premature_image(mask_path):
    #         print(i, mask_path)

if __name__ == "__main__":
    main()