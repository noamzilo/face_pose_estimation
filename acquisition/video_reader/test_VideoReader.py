
from pytest import fixture
import numpy as np
import sys
import os
from code.acquisition.video_reader.VideoReader import VideoReader
import cv2


@fixture()
def video_path():
    cwd = os.getcwd()
    # path = r"data/videos/one_woman_occlusion.mp4"
    path = r"C:\noam\face_pose_estimation\data\videos\one_woman_occlusion.mp4"
    assert os.path.isfile(path)
    return path


@fixture()
def video_reader(video_path):
    reader = VideoReader(video_path)
    return reader


def test_video_reader(video_reader):
    for frame in video_reader.frames():
        cv2.imshow(f'Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break



    cv2.destroyAllWindows()