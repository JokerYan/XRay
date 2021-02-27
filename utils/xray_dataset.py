import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.image_preprocessing import load_video_paths, get_video_frame_count, get_frame_from_video, get_blank_mask_from_size
from utils.video_to_csv import load_from_csv

bucket_frame_count = 5000

class XRayDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.tranform = transform
        self.video_mask_list = load_from_csv(csv_path)

        # # calculate length
        # self.video_data_dict = {}
        # self.length = 0
        # for path_pair in video_mask_list:
        #     video_path, mask_path, is_fake = path_pair
        #     video_frame_count = get_video_frame_count(video_path)
        #     video_data = VideoData(video_path, mask_path, is_fake, self.length, video_frame_count)
        #     start_bucket = self.length // bucket_frame_count
        #     self.length += video_frame_count
        #     end_bucket = self.length // bucket_frame_count
        #
        #     if start_bucket not in self.video_data_dict:
        #         self.video_data_dict[start_bucket] = [video_data]
        #     elif video_data not in self.video_data_dict[start_bucket]:
        #         self.video_data_dict[start_bucket].append(video_data)
        #     if end_bucket not in self.video_data_dict:
        #         self.video_data_dict[end_bucket] = [video_data]
        #     elif video_data not in self.video_data_dict[end_bucket]:
        #         self.video_data_dict[end_bucket].append(video_data)
        # self.length = int(self.length)
        self.length = len(self.video_mask_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # bucket = idx // bucket_frame_count
        # item = None
        # for video_data in self.video_data_dict[bucket]:
        #     if video_data.match(idx):
        #         data = video_data.get_frames_from_idx(idx)
        #         item = {"video_frame": data[0], "mask_frame": data[1], 'is_fake': data[2]}
        #         break
        # assert item is not None
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


# class VideoData:
#     def __init__(self, video_path, mask_path, is_fake, start_idx, frame_count):
#         self.video_path = video_path
#         self.mask_path = mask_path
#         self.start_idx = start_idx
#         self.frame_count = frame_count
#         self.is_fake = is_fake
#
#     def match(self, idx):
#         if idx >= self.start_idx and idx < self.start_idx + self.frame_count:
#             return True
#         return False
#
#     def get_frames_from_idx(self, idx):
#         frame_idx = idx - self.start_idx
#         video_frame = get_frame_from_video(self.video_path, frame_idx)
#         if self.mask_path is None:
#             size = video_frame.shape[:2]  # H x W x C -> H x W
#             mask_frame = get_blank_mask_from_size(size)
#         else:
#             mask_frame = get_frame_from_video(self.mask_path, frame_idx, gray_scale=True)
#         return video_frame, mask_frame, self.is_fake