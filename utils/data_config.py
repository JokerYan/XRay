import os
from sys import platform

if platform == 'win32':
    data_root_dir = r"C:/Code/Projects/XRay/Data/"
elif platform == 'linux':
    data_root_dir = os.path.expanduser('~/zhiwen/XRay/Data/')
else:
    data_root_dir = r''

# video to csv
train_val_ratio = 0.85
data_csv_folder = "./data/"
train_video_csv_path = os.path.join(data_csv_folder, "train_video.csv")
val_video_csv_path = os.path.join(data_csv_folder, "val_video.csv")

# video to image
skip_frame = 5
image_output_dir = os.path.join(data_root_dir, "FaceForensicImage/")
train_image_csv_path = os.path.join(data_csv_folder, "train_image.csv")
val_image_csv_path = os.path.join(data_csv_folder, "val_image.csv")
image_save_size = int(256 / 0.875)

# video/image preprocessing
fake_root_dir = os.path.join(data_root_dir, "FaceForensic/manipulated_sequences/")
fake_root_dir_with_dataset = os.path.join(fake_root_dir, "**/")
video_folder = "c23/videos/"
mask_folder = "masks/videos/"
fake_video_dir = os.path.join(fake_root_dir_with_dataset, video_folder)
fake_mask_dir = os.path.join(fake_root_dir_with_dataset, mask_folder)
real_root_dir = os.path.join(data_root_dir, r"FaceForensic/original_sequences/")
real_video_dir = os.path.join(real_root_dir, "**/")

# select/balance data
# dataset_name = 'Deepfakes'
dataset_name = 'Face2Face'
train_image_selected_csv_path = os.path.join(data_csv_folder, "train_image_selected.csv")
val_image_selected_csv_path = os.path.join(data_csv_folder, "val_image_selected.csv")