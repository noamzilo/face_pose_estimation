import torch
from facenet_pytorch import MTCNN
from src.Utils.ConfigProvider import ConfigProvider
import numpy as np
import cv2
from src.post_processing.PostProcessor import PostProcessor
from src.statefull_processing.BboxTracker import BboxTracker


class StatefulFrameProcessor(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on device: {self._device}')
        self._mtcnn = MTCNN(keep_all=True, device=self._device)
        self._bbox_tracker = BboxTracker()

        # self._last_frame_filtered_bboxes, self._last_frame_confidences = None, None

    def process_single_frame(self, frame, frame_index):
        detected_bboxes, confidences = self._mtcnn.detect(frame)
        detected_bboxes = [] if detected_bboxes is None else detected_bboxes
        confidences = [] if confidences is None else confidences

        confident_bboxes = np.array([bbox for bbox, confidence in zip(detected_bboxes, confidences) if
                                     self._config.confidence_threshold < confidence])
        tracked_bboxes = self._bbox_tracker.update_tracked_bboxes(
            frame_index=frame_index,
            new_bboxes=confident_bboxes)

        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        if tracked_bboxes is not None:
            frame = PostProcessor.draw_rectengles(frame, tracked_bboxes)
            frame = PostProcessor.blur_at_bboxes(frame, tracked_bboxes)
        return frame
