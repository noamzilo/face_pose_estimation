from src.pipeline.example import example_pipeline
from src.Utils.ConfigParser import ConfigParser


if __name__ == "__main__":
    config_path = r"C:\noam\face_pose_estimation\src\config\config.yaml"
    config = ConfigParser(config_path).parse()
    example_pipeline(config.data.path_to_video)  # change this line to be called via web-api
