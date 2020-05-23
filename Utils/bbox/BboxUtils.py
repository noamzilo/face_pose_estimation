import numpy as np


class BboxUtils(object):
    @staticmethod
    def ious(bboxes1, bboxes2):
        left1, top1, right1, bottom1 = BboxUtils.bbox_to_perimiters(bboxes1)
        left2, top2, right2, bottom2 = BboxUtils.bbox_to_perimiters(bboxes2)

        area1 = (right1 - left1) * (top1 - bottom1)
        area2 = (right2 - left2) * (top2 - bottom2)

        intersection_left = np.maximum(left1, left2)
        intersection_right = np.minimum(right1, right2)
        intersection_top = np.maximum(top1, top2)
        intersection_bottom = np.minimum(bottom1, bottom2)
        intersection_area = (intersection_right - intersection_left) * (intersection_top - intersection_bottom)
        intersection_area[intersection_area < 0] = 0

        iou = intersection_area / (area1 + area2 - intersection_area)

        return iou

    @staticmethod
    def bbox_to_perimiters(bboxes):
        left, w, top, h = np.split(bboxes.reshape(-1), 4)
        right = left + w
        bottom = top + h
        return left, right, top, bottom
