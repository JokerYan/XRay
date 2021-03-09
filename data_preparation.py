import os
import random

from utils.data_config import data_csv_folder, train_video_csv_path, val_video_csv_path, fake_video_dir, \
    fake_mask_dir, real_video_dir, train_val_ratio, image_output_dir, train_image_csv_path, val_image_csv_path, \
    train_image_selected_csv_path, val_image_selected_csv_path, dataset_name
from utils.video_image_converter import save_all_frames_from_csv
from utils.video_to_csv import load_video_paths, write_to_csv, balance_data


def collect_video_to_csv():
    all_fake_video_paths = load_video_paths(fake_video_dir, fake_mask_dir)
    all_real_video_paths = load_video_paths(real_video_dir, None)
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
    save_all_frames_from_csv(train_video_csv_path, train_image_csv_path)
    save_all_frames_from_csv(val_video_csv_path, val_image_csv_path)


def select_data():
    train_data_selected = balance_data(train_image_csv_path, ratio=0.5, dataset=dataset_name)
    val_data_selected = balance_data(val_image_csv_path, ratio=0.5, dataset=dataset_name)
    with open(train_image_selected_csv_path, "w+") as csv_file:
        csv_file.write("\n".join(train_data_selected))
    with open(val_image_selected_csv_path, "w+") as csv_file:
        csv_file.write("\n".join(val_data_selected))


def main():
    collect_video_to_csv()
    save_frame_from_video()
    select_data()


if __name__ == '__main__':
    main()