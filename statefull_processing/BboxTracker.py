from src.Utils.ConfigProvider import ConfigProvider
from src.Utils.bbox.BboxUtils import BboxUtils


class BboxTracker(object):
    def __init__(self, ):
        self._config = ConfigProvider.config()

        self._last_frame_bboxes = None

    def update_tracked_bboxes(self, frame_index, new_bboxes):
        if self._last_frame_bboxes is None:
            self._last_frame_bboxes = new_bboxes
            return new_bboxes

        self._track_with_same_place_iou(frame_index, new_bboxes)
        same_place_tracking = self._track_with_same_place_bboxes(new_bboxes)
        return same_place_tracking

    def _track_with_same_place_bboxes(self, new_bboxes):
        if len(new_bboxes) > 0:
            tracked_bboxes = new_bboxes
        else:  # most naiive, just copy from last frame
            tracked_bboxes = self._last_frame_bboxes
        self._last_frame_bboxes = tracked_bboxes

        return tracked_bboxes

    def _track_with_same_place_iou(self, frame_index, new_bboxes):
        # ious = BboxUtils.ious_same_ids(self._last_frame_bboxes, new_bboxes)
        ious = BboxUtils.ious(self._last_frame_bboxes, new_bboxes)
        pass