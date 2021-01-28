import os
import random
import numpy as np

from utils.image_preprocessing import load_video_paths, get_video_frame_count
from utils.image_preprocessing import default_video_dir, default_mask_dir, default_real_video_dir

train_val_ratio = 0.7
data_folder = "./data/"
train_out_path = os.path.join(data_folder, "train_data.csv")
val_out_path = os.path.join(data_folder, "val_data.csv")

def write_to_csv(csv_path, video_paths):
    print("writing video paths to {}".format(csv_path))
    lines = []
    for (video_path, mask_path) in video_paths:
        is_fake = "1" if mask_path is not None else "0"
        mask_path = "" if mask_path is None else mask_path
        line = ",".join((video_path, mask_path, is_fake))
        lines.append(line)
    with open(csv_path, "w+") as out_file:
        out_file.write("\n".join(lines))


def load_from_csv(csv_path):
    out_list = []
    with open(csv_path, "r") as csv_file:
        for line in csv_file:
            line_parts = line.strip().split(",")
            video_path = line_parts[0]
            mask_path = line_parts[1] if line_parts[1] else None
            is_fake = np.float32(line_parts[2])
            out_list.append([video_path, mask_path, is_fake])
    return out_list


def main():
    all_fake_video_paths = load_video_paths(default_video_dir, default_mask_dir)
    all_real_video_paths = load_video_paths(default_real_video_dir, None)
    all_video_paths = all_fake_video_paths + all_real_video_paths
    random.shuffle(all_video_paths)
    train_count = int(len(all_video_paths) * train_val_ratio)

    train_video_paths = all_video_paths[:train_count]
    val_video_paths = all_video_paths[train_count:]
    print(len(train_video_paths), len(val_video_paths))

    os.makedirs(data_folder, exist_ok=True)
    write_to_csv(train_out_path, train_video_paths)
    write_to_csv(val_out_path, val_video_paths)

if __name__ == "__main__":
    main()