from src.Utils.ConfigProvider import ConfigProvider
from src.acquisition.video_reader.VideoReader import VideoReader
import cv2
from src.statefull_processing.StatefulFrameProcessor import StatefulFrameProcessor


class Pipeline(object):
    def __init__(self):
        self._config = ConfigProvider.config()
        self._video_reader = VideoReader(
            path_to_video=self._config.data.path_to_video,
            mode='PIL',
            downsample_factor=0.25)
        self._frame_processor = StatefulFrameProcessor()

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

    def example_pipeline(self):
        frames = self._video_reader.frames(start=self._start_frame, end=self._end_frame, )

        frames_tracked = []
        for i, frame in enumerate(frames):
            self._current_frame_id += 1
            print(f'\rTracking frame: {self._current_frame_id}', end='')
            processed_frame = self._frame_processor.process_single_frame(
                frame=frame,
                frame_index=i)

            cv2.imshow(f'Frame', processed_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frames_tracked.append(processed_frame)
        print(f'\nDone')

        for frame in frames_tracked:
            self._video_writer.write(frame)
        self._video_writer.release()
