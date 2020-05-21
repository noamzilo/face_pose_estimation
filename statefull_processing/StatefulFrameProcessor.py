import torch
from facenet_pytorch import MTCNN
from src.Utils.ConfigProvider import ConfigProvider
import numpy as np
import cv2
from src.post_processing.PostProcessor import PostProcessor


class StatefulFrameProcessor(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self._device}')
        self._mtcnn = MTCNN(keep_all=True, device=self._device)

        self._last_frame_filtered_bboxes, self._last_frame_confidences = None, None

    def process_single_frame(self, frame):
        bboxes, confidences = self._mtcnn.detect(frame)

        if bboxes is not None:
            filtered_bboxes = [bbox for bbox, confidence in zip(bboxes, confidences) if
                               self._config.confidence_threshold < confidence]
            if len(filtered_bboxes) == 0:
                filtered_bboxes = self._last_frame_filtered_bboxes
        else:  # most naiive, just copy from last frame
            filtered_bboxes = self._last_frame_filtered_bboxes
        self._last_frame_filtered_bboxes, self._last_frame_confidences = filtered_bboxes, confidences
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        if bboxes is not None:
            frame = PostProcessor.draw_rectengles(frame, bboxes)
            frame = PostProcessor.blur_at_bboxes(frame, bboxes)
        return frame
