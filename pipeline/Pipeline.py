from src.Utils.ConfigProvider import ConfigProvider
import torch
from facenet_pytorch import MTCNN
from src.acquisition.video_reader.VideoReader import VideoReader
from src.post_processing.PostProcessor import PostProcessor
import numpy as np
import cv2


class Pipeline(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mtcnn = MTCNN(keep_all=True, device=self._device)
        self._video_reader = VideoReader(
            path_to_video=self._config.data.path_to_video,
            mode='PIL',
            downsample_factor=0.25)

        self._video_writer = self._create_video_writer()

        self._start_frame, self._end_frame = 0, 200

        self._last_frame_filtered_bboxes, self._last_frame_confidences = None, None

        self._current_frame_id = 0

    def _create_video_writer(self):
        video_writer_fourrcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_tracked = cv2.VideoWriter(
            self._config.output.path_to_output_file,
            video_writer_fourrcc,
            self._video_reader.frame_rate,
            self._video_reader.shape[:2])
        return video_tracked

    def process_single_frame(self, frame):
        bboxes, confidences = self._mtcnn.detect(frame)

        if bboxes is not None:
            filtered_bboxes = [bbox for bbox, confidence in zip(bboxes, confidences) if
                               self._config.confidence_threshold < confidence]
            if len(filtered_bboxes) == 0:
                filtered_bboxes = self._last_frame_filtered_bboxes
        else:  # most naiive, just copy from last frame
            filtered_bboxes = self._last_frame_filtered_bboxes
        self._last_frame_filtered_bboxes, self._last_frame_confidences = filtered_bboxes, confidences
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frame = self._post_process_frame(frame, filtered_bboxes)
        return frame

    def example_pipeline(self):
        print(f'Running on device: {self._device}')

        frames = self._video_reader.frames(start=self._start_frame, end=self._end_frame, )

        frames_tracked = []
        for i, frame in enumerate(frames):
            self._current_frame_id += 1
            print(f'\rTracking frame: {self._current_frame_id}', end='')
            processed_frame = self.process_single_frame(frame)

            cv2.imshow(f'Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frames_tracked.append(processed_frame)
        print(f'\nDone')

        for frame in frames_tracked:
            self._video_writer.write(frame)
        self._video_writer.release()

    @staticmethod
    def _post_process_frame(frame, bboxes):
        if bboxes is not None:
            frame = PostProcessor.draw_rectengles(frame, bboxes)
            frame = PostProcessor.blur_at_bboxes(frame, bboxes)
        return frame
