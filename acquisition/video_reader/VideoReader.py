import cv2


class VideoReader(object):
    def __init__(self, path_to_video):
        self._path_to_video = path_to_video

        self._create_capture_object()

    def _create_capture_object(self):
        self._cap = cv2.VideoCapture(self._path_to_video)

    def frames(self):
        cap = self._cap
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
