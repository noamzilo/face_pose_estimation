import cv2
from PIL import ImageDraw


class PostProcessor(object):
    confidence_threshold = 0.95
    bbox_color = (255, 0, 0)
    bbox_width = 6

    @staticmethod
    def draw_rectengles(frame, bboxes, confidences):
        if bboxes is None:
            raise ValueError("no bboxes to draw")
        assert len(bboxes) == len(confidences)

        draw = ImageDraw.Draw(frame)
        if bboxes is None:
            raise ValueError("no bboxes to draw")
        for box, confidence in zip(bboxes, confidences):
            if PostProcessor.confidence_threshold < confidence:
                draw.rectangle(box.tolist(), outline=PostProcessor.bbox_color, width=PostProcessor.bbox_width)
        return frame

