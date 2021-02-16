import os
import random
import numpy as np
from glob import glob

from utils.image_preprocessing import load_video_paths, get_video_frame_count
from utils.image_preprocessing import default_video_dir, default_mask_dir, default_real_video_dir

# train_val_ratio = 0.7
# data_folder = "./data/"
# train_out_path = os.path.join(data_folder, "train_video.csv")
# val_out_path = os.path.join(data_folder, "val_video.csv")

train_val_ratio = 0.85
data_folder = "./data/"
train_out_path = os.path.join(data_folder, "train_video.csv")
val_out_path = os.path.join(data_folder, "val_video.csv")

def write_to_csv(csv_path, video_paths):
    print("writing paths to {}".format(csv_path))
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

def concate_csv(csv_path1, csv_path2, csv_path_output):
    with open(csv_path1, 'r') as csv_file:
        csv_content_1 = csv_file.read()
    with open(csv_path2, 'r') as csv_file:
        csv_content_2 = csv_file.read()
    if not csv_content_1.endswith('\n'):
        csv_content_1 += "\n"
    csv_content_output = csv_content_1 + csv_content_2
    with open(csv_path_output, 'w+') as csv_file:
        csv_file.write(csv_content_output)


def get_pos_neg_data(csv_path):
    with open(csv_path, 'r') as csv_file:
        lines = csv_file.readlines()
    pos_lines = []
    neg_lines = []
    for line in lines:
        if line.strip().split(",")[-1] == "1":
            pos_lines.append(line.strip())
        else:
            neg_lines.append(line.strip())
    pos_count = len(pos_lines)
    neg_count = len(neg_lines)
    total_count = len(lines)
    pos_ratio = pos_count / total_count * 100
    print("Data Statistics from: {}".format(csv_path))
    print("Total: {0}\tPos: {1}\tNeg: {2}\tPos Ratio: {3:.2f}%".format(total_count, pos_count, neg_count, pos_ratio))
    return pos_lines, neg_lines

def filter_dataset(fake_data_lines, dataset):
    filtered_lines = []
    for line in fake_data_lines:
        if not dataset or dataset in line:
            filtered_lines.append(line)
    return filtered_lines

def balance_data(csv_path, ratio=0.5, dataset=''):
    pos_neg_data = get_pos_neg_data(csv_path)
    pos_data, neg_data = pos_neg_data
    pos_data = filter_dataset(pos_data, dataset)
    pos_count = len(pos_data)
    neg_count = len(neg_data)
    total_count = min(pos_count / ratio, neg_count / (1 - ratio))
    pos_target_count = int(total_count * ratio)
    neg_target_count = int(total_count * (1 - ratio))
    random.shuffle(pos_data)
    random.shuffle(neg_data)
    pos_data_selected = pos_data[:pos_target_count]
    neg_data_selected = neg_data[:neg_target_count]
    print("Balanced: Pos {}    Neg {}".format(pos_target_count, neg_target_count))
    return pos_data_selected + neg_data_selected

def main():
    # all_fake_video_paths = load_video_paths(default_video_dir, default_mask_dir)
    # all_real_video_paths = load_video_paths(default_real_video_dir, None)
    # all_video_paths = all_fake_video_paths + all_real_video_paths
    # random.shuffle(all_video_paths)
    # train_count = int(len(all_video_paths) * train_val_ratio)
    #
    # train_video_paths = all_video_paths[:train_count]
    # val_video_paths = all_video_paths[train_count:]
    # print(len(train_video_paths), len(val_video_paths))
    #
    # os.makedirs(data_folder, exist_ok=True)
    # write_to_csv(train_out_path, train_video_paths)
    # write_to_csv(val_out_path, val_video_paths)

    # concate
    # concate_csv("./data/train_image.csv", "./data/train_image_youtube.csv", "./data/train_image_full.csv")
    # concate_csv("./data/val_image.csv", "./data/val_image_youtube.csv", "./data/val_image_full.csv")

    train_data_selected = balance_data("./data/train_image.csv", ratio=0.5, dataset='Face2Face')
    val_data_selected = balance_data("./data/val_image.csv", dataset='Face2Face')
    with open("./data/train_image_selected.csv", "w+") as csv_file:
        csv_file.write("\n".join(train_data_selected))
    with open("./data/val_image_selected.csv", "w+") as csv_file:
        csv_file.write("\n".join(val_data_selected))

if __name__ == "__main__":
    main()