import numpy as np


class BboxUtils(object):
    @staticmethod
    def ious(bboxes1, bboxes2):
        # can probably be optimized by making it less readable using algebra.
        left1, top1, right1, bottom1 = np.split(bboxes1)
        left2, top2, right2, bottom2 = np.split(bboxes2)

        area1 = (right1 - left1) * (top1 - bottom1)
        area2 = (right2 - left2) * (top2 - bottom2)

        # union_left = np.minimum(left1, left2)
        # union_right = np.maximum(right1, right2)
        # union_top = np.minimum(top1, top2)
        # union_bottom = np.maximum(bottom1, bottom2)
        # union_area = (union_right - union_left) * (union_top - union_bottom)
        # union_area[union_area < 0] = 0

        intersection_left = np.maximum(left1, left2)
        intersection_right = np.minimum(right1, right2)
        intersection_top = np.maximum(top1, top2)
        intersection_bottom = np.minimum(bottom1, bottom2)
        intersection_area = (intersection_right - intersection_left) * (intersection_top - intersection_bottom)
        intersection_area[intersection_area < 0] = 0

        iou = intersection_area / (area1 + area2 - intersection_area)

        return iou
