
from pytest import fixture
import numpy as np
import sys
import os
from src.acquisition.video_reader.VideoReader import VideoReader
import cv2


@fixture()
def video_path():
    cwd = os.getcwd()
    # path = r"data/videos/one_woman_occlusion.mp4"
    path = r"C:\noam\face_pose_estimation\data\videos\one_woman_occlusion.mp4"
    assert os.path.isfile(path)
    return path


@fixture()
def video_reader_opencv(video_path):
    reader = VideoReader(video_path, mode='opencv')
    return reader


@fixture()
def video_reader_pil(video_path):
    reader = VideoReader(video_path, mode='PIL')
    return reader


def test_video_reader_opencv(video_reader_opencv):
    for frame in video_reader_opencv.frames(start=10, end=100, skip=5):
        cv2.imshow(f'Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def test_video_reader_pil(video_reader_pil):
    for frame in video_reader_pil.frames(start=10, end=100, skip=5):
        cv2_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2.imshow(f'Frame', cv2_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

