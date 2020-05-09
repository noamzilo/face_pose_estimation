from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

from src.acquisition.video_reader.VideoReader import VideoReader

def example_facenet():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    mtcnn = MTCNN(keep_all=True, device=device)

    path_to_video = r"C:\noam\face_pose_estimation\data\videos\one_woman_occlusion.mp4"
    video = mmcv.VideoReader(path_to_video)

    video_reader = VideoReader(path_to_video=path_to_video, mode='PIL')
    frames = video_reader.frames(start=0, end=10)
    # frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]

    display.Video(path_to_video, width=640)

    # run video through MTCNN
    frames_tracked = []
    for i, frame in enumerate(frames):
        print(f'\rTracking frame: {i + 1}', end='')

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
    print(f'\nDone')

    # display detections
    d = display.display(frames_tracked[0], display_id=True)
    i = 1
    try:
        while True:
            d.update(frames_tracked[i % len(frames_tracked)])
            i += 1
    except KeyboardInterrupt:
        pass

    # save tracked video
    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()

example_facenet()
