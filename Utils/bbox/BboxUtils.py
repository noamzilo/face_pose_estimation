import numpy as np


class BboxUtils(object):
    @staticmethod
    def ious_same_ids(bboxes1, bboxes2):
        left1, top1, right1, bottom1 = BboxUtils.bbox_to_perimiters(bboxes1)
        left2, top2, right2, bottom2 = BboxUtils.bbox_to_perimiters(bboxes2)

        area1 = (right1 - left1) * (bottom1 - top1)
        area2 = (right2 - left2) * (bottom2 - top2)

        intersection_left = np.maximum(left1, left2)
        intersection_right = np.minimum(right1, right2)
        intersection_top = np.maximum(top1, top2)
        intersection_bottom = np.minimum(bottom1, bottom2)
        intersection_area = (intersection_right - intersection_left) * (intersection_bottom - intersection_top)
        intersection_area[intersection_area < 0] = 0

        iou = intersection_area / (area1 + area2 - intersection_area)

        return iou

    @staticmethod
    def ious(bboxes1, bboxes2):
        """
        calculate IOUS https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        between all bboxes1 and bboxes2
        :param bboxes1: NON EMPTY list of bboxes, each of the form left, top, right, bottom
        :param bboxes2: NON EMPTY list of bboxes, each of the form left, top, right, bottom
        :return: np.array of shape (bboxes1.shape[0], bboxes2.shape[0]).
                 index i, j contains the IOU between bboxes1[i] and bboxes2[j] if any overlap, else 0.
        """
        # probably can be vectorized, but no real need for <7 matches per frame.
        assert len(bboxes1) > 0
        assert len(bboxes2) > 0

        ious = np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                iou = BboxUtils.ious_same_ids(bbox1, bbox2)
                ious[i, j] = iou

        return ious

    @staticmethod
    def bbox_to_perimiters(bboxes):
        left, top, right, bottom = np.split(bboxes.reshape(-1), 4)
        return left, top, right, bottom
