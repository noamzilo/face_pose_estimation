from src.Utils.ConfigProvider import ConfigProvider
from src.Utils.bbox.BboxUtils import BboxUtils
import numpy as np


class BboxTracker(object):
    def __init__(self, ):
        self._config = ConfigProvider.config()

        self._detections = []

    def update_tracked_bboxes(self, frame_index, new_bboxes):
        if len(self._detections) == 0:
            self._detections = new_bboxes
            return self._detections

        iou_tracking = self._track_with_same_place_iou(new_bboxes)
        self._update_tracking(same_detections_new_)
        return iou_tracking

    def _track_with_same_place_iou(self, new_bboxes):
        """
        calculate remaining, new, and lost boxes based on new_bboxes and the state in self._detections
        :param new_bboxes: np.array of shape (#detections, 4), columns are (left, top, right, bottom) correspondingly
        :return:
        remaining_det_inds: np.array of inds in new_bboxes of detections that were found in the new frame AND in the
                            previous frame
        new_det_inds: np.array of inds in new_bboxes of detections that were found in the new frame but not in the
                      previous one
        lost_det_inds: np.array of inds in self._detections of detections that were found in previous frame but not in
                       the new frame
        """
        # ious = BboxUtils.ious_same_ids(self._last_frame_bboxes, new_bboxes)
        old_dets = np.array(self._detections)
        ious = BboxUtils.ious(old_dets, new_bboxes)
        same_object_inds = np.where(self._config.tracking.iou_threshold < ious)
        same_object_inds_last_frame, same_object_inds_new_frame = same_object_inds

        # new locations of boxes that intersect old boxes
        remaining_det_inds = same_object_inds_new_frame
        # remaining_detections_new_location = new_bboxes[same_object_inds_new_frame, :]

        # new locations of boxes that don't intersect old boxes
        new_det_inds = np.delete(np.arange(new_bboxes.shape[0]), same_object_inds_new_frame)
        # new_detections = new_bboxes[np.delete(np.arange(new_bboxes.shape[0]), same_object_inds_new_frame), :]
        # assert remaining_detections_new_location.shape[0] + new_detections.shape[0] == new_bboxes.shape[0]
        assert len(remaining_det_inds) + len(new_det_inds) == new_bboxes.shape[0]

        # old locations of boxes that don't intersect new boxes
        lost_det_inds = np.delete(np.arange(new_bboxes.shape[0]), same_object_inds_last_frame)
        # lost_detections = old_dets[np.delete(np.arange(new_bboxes.shape[0]), same_object_inds_last_frame), :]

        return remaining_det_inds, new_det_inds, lost_det_inds
