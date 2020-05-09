from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv
import cv2


from src.acquisition.video_reader.VideoReader import VideoReader
from src.post_processing.PostProcessor import PostProcessor


def example_pipeline():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    mtcnn = MTCNN(keep_all=True, device=device)

    path_to_video = r"C:\noam\face_pose_estimation\data\videos\one_woman_occlusion.mp4"

    video_reader = VideoReader(path_to_video=path_to_video, mode='PIL', downsample_factor=0.25)
    frames = video_reader.frames(start=60, end=120,)

    # run video through MTCNN
    frames_tracked = []

    for i, frame in enumerate(frames):
        print(f'\rTracking frame: {i + 1}', end='')

        # Detect faces
        bboxes, confidences = mtcnn.detect(frame)
        confidence_threshold = 0.95

        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        if bboxes is not None:
            filtered_bboxes = [bbox for bbox, confidence in zip(bboxes, confidences) if confidence_threshold < confidence]
            frame = PostProcessor.draw_rectengles(frame, filtered_bboxes)
            frame = PostProcessor.blur_at_bboxes(frame, filtered_bboxes)

        # display detections
        cv2.imshow(f'Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Add to frame list
        frames_tracked.append(frame)

    print(f'\nDone')

    # save tracked video
    out_path = r'C:\noam\face_pose_estimation\output\output.mp4'
    video_writer_fourrcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_tracked = cv2.VideoWriter(out_path, video_writer_fourrcc, video_reader.frame_rate, video_reader.shape[:2])
    for frame in frames_tracked:
        video_tracked.write(frame)
    video_tracked.release()


if __name__ == "__main__":
    example_pipeline()
