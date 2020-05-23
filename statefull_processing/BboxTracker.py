from src.Utils.ConfigProvider import ConfigProvider


class BboxTracker(object):
    def __init__(self, ):
        self._config = ConfigProvider.config()
        self._tracked_bboxes = []
        self._last_frame_filtered_bboxes, self._last_frame_confidences = None, None

    def update_tracked_bboxes(self, new_bboxes, confidences):
        assert new_bboxes is None or (len(new_bboxes) == len(confidences))
        if new_bboxes is not None:
            tracked_bboxes = confident_bboxes = [bbox for bbox, confidence in zip(new_bboxes, confidences) if
                                                 self._config.confidence_threshold < confidence]
            if len(confident_bboxes) == 0:
                tracked_bboxes = self._last_frame_filtered_bboxes
        else:  # most naiive, just copy from last frame
            tracked_bboxes = self._last_frame_filtered_bboxes
        self._last_frame_filtered_bboxes, self._last_frame_confidences = tracked_bboxes, confidences

        return tracked_bboxes
