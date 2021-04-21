import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.image_preprocessing import get_blank_mask_from_size
from utils.video_to_csv import load_from_csv

bucket_frame_count = 5000

class XRayDataset(Dataset):
    def __init__(self, csv_path, transform, target_class=None):
        self.tranform = transform
        self.video_mask_list = load_from_csv(csv_path, target_class)


        self.length = len(self.video_mask_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_frame_path, mask_frame_path, is_fake = self.video_mask_list[idx]
        # print(video_frame_path, mask_frame_path)
        video_frame = np.float32(cv2.imread(video_frame_path))
        if mask_frame_path is not None:
            mask_frame = np.float32(cv2.imread(mask_frame_path))
        else:
            mask_frame = np.float32(get_blank_mask_from_size(video_frame.shape))
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        item = {"video_frame": video_frame, "mask_frame": mask_frame, "is_fake": is_fake}
        if self.tranform:
            item = self.tranform(item)
        return item