import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2 as cv

class C3DDataset(Dataset):
    def __init__(self, root_dir, num_frames):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.video_paths, self.labels = self._get_video_paths_and_labels()

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]

        frames = []
        frame_names = os.listdir(video_path)
        frame_indices = self._get_frame_indices(frame_names)
        for frame_index in frame_indices:
            frame_name = frame_names[frame_index]
            frame_path = os.path.join(video_path, frame_name)
            frame = self._load_frame(frame_path)
            frames.append(frame)

        video_data = torch.from_numpy(np.array(frames)).permute(3, 0, 1 ,2)

        return video_data, label

    def _get_video_paths_and_labels(self):
        video_paths = []
        labels = []
        actions = os.listdir(self.root_dir)
        for label, action in enumerate(actions):
            action_dir = os.path.join(self.root_dir, action)
            if not os.path.isdir(action_dir):
                continue
            for video_name in os.listdir(action_dir):
                video_path = os.path.join(action_dir, video_name)
                video_paths.append(video_path)
                labels.append(label)
        return video_paths, labels

    def _get_frame_indices(self, frame_names):
        total_frames = len(frame_names)
        if total_frames < self.num_frames:
            indices = list(range(total_frames))
            indices += [indices[-1]] * (self.num_frames - total_frames)  # 重复最后一帧来填充不足的帧数
        else:
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
        return indices

    def _load_frame(self, frame_path):
        frame = cv.imread(frame_path).astype(np.float32)
        return frame
