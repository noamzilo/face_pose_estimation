from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

def example_facenet():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    mtcnn = MTCNN(keep_all=True, device=device)

    video_path = r"C:\noam\face_pose_estimation\data\videos\one_woman_occlusion.mp4"
    video = mmcv.VideoReader(video_path)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

    display.Video(video_path, width=640)

    frames_tracked = []
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')

        # Detect faces
        boxes, _ = mtcnn.detect(frame)

        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        if boxes is None:
            continue
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        # Add to frame list
        frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    print('\nDone')

example_facenet()
