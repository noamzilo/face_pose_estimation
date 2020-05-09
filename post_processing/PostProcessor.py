import cv2
from PIL import ImageDraw
from PIL import ImageFilter
import numpy as np


class PostProcessor(object):

    bbox_color = (255, 0, 0)
    bbox_width = 6

    @staticmethod
    def draw_rectengles(frame_pil, bboxes):
        if bboxes is None:
            return frame_pil

        draw = ImageDraw.Draw(frame_pil)
        for bbox in bboxes:
            draw.rectangle(bbox.tolist(), outline=PostProcessor.bbox_color, width=PostProcessor.bbox_width)
        return frame_pil

    @staticmethod
    def blur_at_bboxes(frame_cv2, bboxes):
        if bboxes is None:
            return frame_cv2

        for bbox in np.array(bboxes, dtype=np.int):
            left = bbox[0]
            top = bbox[1]
            right = left + bbox[2]
            bottom = top + bbox[3]

            frame_cv2[top:bottom, left:right] = \
                cv2.GaussianBlur(src=frame_cv2[top:bottom, left:right], ksize=(51, 51), sigmaX=0, sigmaY=0)
        return frame_cv2
