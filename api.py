from src.pipeline.example import example_pipeline
from src.Utils.ConfigParser import ConfigParser
from src.pipeline.Pipeline import Pipeline


if __name__ == "__main__":
    # example_pipeline(config.data.path_to_video)  # change this line to be called via web-api
    pipeline = Pipeline()
    pipeline.example_pipeline()
