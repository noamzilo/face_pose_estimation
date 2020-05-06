import cv2


class VideoReader(object):
    def __init__(self, path_to_video):
        self._path_to_video = path_to_video

        self._create_capture_object()

    def _create_capture_object(self):
        self._cap = cv2.VideoCapture(self._path_to_video)

    def frames(self, start, end, skip=None):
        if skip is not None:
            raise NotImplementedError

        current_frame_ind = 0
        cap = self._cap
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame_ind < start:
                continue
            elif end <= current_frame_ind:
                break
            yield frame
            current_frame_ind += 1
