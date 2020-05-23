from src.Utils.ConfigProvider import ConfigProvider
from src.Utils.bbox.BboxUtils import BboxUtils


class BboxTracker(object):
    def __init__(self, ):
        self._config = ConfigProvider.config()

        self._tracked_bboxes = []
        self._last_frame_bboxes = None

    def update_tracked_bboxes(self, frame_index, new_bboxes):
        return self._track_with_same_place_bboxes(frame_index, new_bboxes)

    def _track_with_same_place_bboxes(self, frame_index, new_bboxes):
        if len(new_bboxes) > 0:
            tracked_bboxes = new_bboxes
        else:  # most naiive, just copy from last frame
            tracked_bboxes = self._last_frame_bboxes
        self._last_frame_bboxes = tracked_bboxes

        return tracked_bboxes

    def _track_with_same_place_iou(self):
        pass