import threading
import queue
import cv2 as cv
import model
import torch
import numpy as np

rtmp_url = ''
weight_path = ''
frequence = 4
num_frames = 16
num_classes = 101
class_names_path = 'your_data/class_names.txt'

with open(class_names_path, 'r') as f:
    class_names = f.readlines()
    f.close()

model = model.C3D(num_classes=num_classes, weight_path=weight_path).cuda()
model.eval()

# 创建线程安全的帧缓冲区队列
frame_queue = queue.Queue(maxsize=num_frames)


# 定义帧缓冲队列处理函数
def process_frames(frames):
    frames = torch.from_numpy(np.array(frames).astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = model.forward(frames)

    probs = torch.nn.Softmax(dim=1)(outputs)
    label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
    print(class_names[label].split(' ')[-1].strip())
    pass


def main_thread():

    video = cv.VideoCapture(rtmp_url)

    frame_count = 0

    while True:

        ret, frame = video.read()

        if not ret:
            break

        # cv.imshow('result', frame)

        frame = cv.resize(frame, (112, 112))

        cv.imshow('result', frame)
        cv.waitKey(1)  # 更新窗口

        frame_count += 1

        if frame_count % frequence == 0:
            frame_count = 0
            frame_queue.put(frame)
            print(frame)


display_thread = threading.Thread(target=main_thread)
display_thread.start()


# 帧处理线程函数
def frame_processing_thread():
    accumulated_frames = []

    while True:
        frame = frame_queue.get()  # 从队列中获取帧
        accumulated_frames.append(frame)

        if len(accumulated_frames) == num_frames:
            process_frames(accumulated_frames)
            accumulated_frames.pop(0)


processing_thread = threading.Thread(target=frame_processing_thread)
processing_thread.start()
