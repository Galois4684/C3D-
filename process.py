import cv2 as cv
import os
from sklearn.model_selection import train_test_split

root = 'your_data/raw_data'
save_root = 'your_data/processed_data'
train_save_dir = os.path.join(save_root, 'train')
val_save_dir = os.path.join(save_root, 'val')
test_size = 0.2

if not os.path.exists(save_root):
    os.makedirs(save_root)

if not os.path.exists(train_save_dir):
    os.makedirs(train_save_dir)

if not os.path.exists(val_save_dir):
    os.makedirs(val_save_dir)

actions = os.listdir(root)
for action in actions:

    if not os.path.exists(os.path.join(train_save_dir, action)):
        os.makedirs(os.path.join(train_save_dir, action))

    if not os.path.exists(os.path.join(val_save_dir, action)):
        os.makedirs(os.path.join(val_save_dir, action))

    videos = os.listdir(os.path.join(root, action))

    # 划分训练集和测试集
    train_videos, val_videos = train_test_split(videos, test_size=test_size, random_state=42)

    for video_name in videos:

        video_path = os.path.join(root, action, video_name)

        if video_name in train_videos:
            save_path = os.path.join(train_save_dir, action, video_name)
        else:
            save_path = os.path.join(val_save_dir, action, video_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        video = cv.VideoCapture(video_path)

        count = 0

        while True:

            ret, frame = video.read()

            if not ret:
                break

            frame = cv.resize(frame, (112, 112))

            cv.imwrite(os.path.join(save_path, f'frame{count}.jpg'), frame)

            print(video_name + f' frame {count} process finished')

            count += 1
