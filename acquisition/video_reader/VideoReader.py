import cv2
import numpy as np
from PIL import Image


class VideoReader(object):
    def __init__(self, path_to_video, mode='opencv', resize_to_shape=None, downsample_factor=1.0):
        self._path_to_video = path_to_video
        self._set_mode(mode)

        self._create_capture_object()
        self._set_resize_to_shape(resize_to_shape, downsample_factor)

    def _create_capture_object(self):
        self._cap = cv2.VideoCapture(self._path_to_video)
        self._width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self._height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._frame_rate = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self._n_channels = 3  # https://stackoverflow.com/questions/61699391/how-to-know-how-many-color-channels-in-cv2-videocapture-object

    def _set_resize_to_shape(self, resize_to_shape, downsample_factor):
        assert resize_to_shape is None or downsample_factor == 1.0
        if resize_to_shape is not None:
            self._output_frame_shape = (resize_to_shape[0], resize_to_shape[1], self._n_channels)
        else:
            frame_shape = tuple(np.array([self._width, self._height]) * downsample_factor)
            self._output_frame_shape = (frame_shape[0], frame_shape[1], self._n_channels)

    def _set_mode(self, mode):
        self._modes = {'opencv', 'PIL'}
        assert mode in self._modes
        self._mode = mode

    def frames(self, start=0, end=np.inf, skip=1):  # generator object for frames
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
                yield self._apply_filters(frame)
            else:
                continue

    def _resize_to_shape(self, frame):
        if self._output_frame_shape is None:
            return frame
        frame = cv2.resize(frame, self._output_frame_shape)
        return frame

    def _apply_filters(self, frame):
        frame = self._resize_to_shape(frame)
        frame = self._convert_np_frame_by_mode(frame)
        return frame

    def _convert_np_frame_by_mode(self, frame):
        if self._mode == 'opencv':
            return frame
        elif self._mode == 'PIL':
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @property
    def original_shape(self):
        return self._height, self._width

    @property
    def shape(self):
        return self._output_frame_shape

    @property
    def frame_rate(self):
        return self._frame_rate

    @property
    def frame_count(self):
        return self._frame_count
