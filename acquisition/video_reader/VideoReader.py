import cv2
import numpy as np
from PIL import Image


class VideoReader(object):
    def __init__(self, path_to_video, mode='opencv'):
        self._path_to_video = path_to_video
        self._set_mode(mode)

        self._create_capture_object()

    def _create_capture_object(self):
        self._cap = cv2.VideoCapture(self._path_to_video)

    def _set_mode(self, mode):
        self._modes = {'opencv', 'PIL'}
        assert mode in self._modes
        self._mode = mode

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
                yield self._convert_np_frame_by_mode(frame)
            else:
                continue

    def _convert_np_frame_by_mode(self, frame):
        if self._mode == 'opencv':
            return frame
        elif self._mode == 'PIL':
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

