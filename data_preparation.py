import os
import random

from utils.data_config import data_csv_folder, train_video_csv_path, val_video_csv_path, data_video_dir, \
    data_mask_dir, data_real_video_dir, train_val_ratio, image_output_dir, train_output_csv_path, val_output_csv_path
from utils.video_image_converter import save_all_frames_from_csv
from utils.video_to_csv import load_video_paths, write_to_csv


def collect_video_to_csv():
    all_fake_video_paths = load_video_paths(data_video_dir, data_mask_dir)
    all_real_video_paths = load_video_paths(data_real_video_dir, None)
    all_video_paths = all_fake_video_paths + all_real_video_paths
    random.shuffle(all_video_paths)
    train_count = int(len(all_video_paths) * train_val_ratio)

    train_video_paths = all_video_paths[:train_count]
    val_video_paths = all_video_paths[train_count:]

    os.makedirs(data_csv_folder, exist_ok=True)
    write_to_csv(train_video_csv_path, train_video_paths)
    write_to_csv(val_video_csv_path, val_video_paths)


def save_frame_from_video():
    os.makedirs(image_output_dir, exist_ok=True)
    save_all_frames_from_csv(train_video_csv_path, train_output_csv_path)
    save_all_frames_from_csv(val_video_csv_path, val_output_csv_path)


def main():
    collect_video_to_csv()
    save_frame_from_video()


if __name__ == '__main__':
    main()