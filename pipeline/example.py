from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw

from src.acquisition.video_reader.VideoReader import VideoReader


def example_facenet():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    mtcnn = MTCNN(keep_all=True, device=device)

    path_to_video = r"C:\noam\face_pose_estimation\data\videos\one_woman_occlusion.mp4"

    video_reader = VideoReader(path_to_video=path_to_video, mode='PIL')
    frames = video_reader.frames(start=0, end=5)
    # frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]


    # run video through MTCNN
    frames_tracked = []
    confidence_threshold = 0.95
    for i, frame in enumerate(frames):
        print(f'\rTracking frame: {i + 1}', end='')

        # Detect faces
        boxes, confidences = mtcnn.detect(frame)

        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        if boxes is None:
            continue
        for box, confidence in zip(boxes, confidences):
            if confidence_threshold < confidence:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        # display detections
        cv2_frame = cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'Frame', cv2_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Add to frame list
        frames_tracked.append(cv2_frame)

    print(f'\nDone')

    # save tracked video
    out_path = r'C:\noam\face_pose_estimation\output\video_tracked.mp4'
    video_writer_fourrcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_tracked = cv2.VideoWriter(out_path, video_writer_fourrcc, video_reader.frame_rate , video_reader.shape[:2])
    for frame in frames_tracked:
        video_tracked.write(frame)
    video_tracked.release()

example_facenet()
