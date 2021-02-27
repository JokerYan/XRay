import os

# video to csv
train_val_ratio = 0.85
data_csv_folder = "./data/"
train_video_csv_path = os.path.join(data_csv_folder, "train_video.csv")
val_video_csv_path = os.path.join(data_csv_folder, "val_video.csv")

# video to image
image_output_dir = r"C:/Code/Projects/XRay/Data/FaceForensicImage/"
train_output_csv_path = os.path.join(data_csv_folder, "train_image.csv")
val_output_csv_path = os.path.join(data_csv_folder, "val_image.csv")
image_save_size = int(256 / 0.875)

# video/image preprocessing
data_root_dir = "C:/Code/Projects/XRay/Data/FaceForensic/manipulated_sequences/"
data_root_dir_with_dataset = data_root_dir + "**/"
data_video_folder = "c23/videos/"
data_mask_folder = "masks/videos/"
data_video_dir = os.path.join(data_root_dir_with_dataset, data_video_folder)
data_mask_dir = os.path.join(data_root_dir_with_dataset, data_mask_folder)
data_real_root_dir = r"C:/Code/Projects/XRay/Data/FaceForensic/original_sequences/"
data_real_video_dir = os.path.join(data_real_root_dir, "**/")