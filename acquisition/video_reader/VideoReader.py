import cv2
import numpy as np


class VideoReader(object):
    def __init__(self, path_to_video):
        self._path_to_video = path_to_video

        self._create_capture_object()

    def _create_capture_object(self):
        self._cap = cv2.VideoCapture(self._path_to_video)

    def frames(self, start=0, end=np.inf, skip=1):
        current_frame_ind = -1
        cap = self._cap
        while cap.isOpened():
            current_frame_ind += 1
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame_ind < start:
                continue
            elif end is None or end <= current_frame_ind:
                break
            if current_frame_ind % skip == 0:
                yield frame
            else:
                continue

