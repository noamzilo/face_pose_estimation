from src.Utils.paths import config_path
from src.Utils.ConfigParser import ConfigParser
from src.Utils.logging.Logging import Logging


class ConfigProvider(object):
    __the_config = None

    @staticmethod
    @Logging.log_in_out
    def config():
        if ConfigProvider.__the_config is None:
            ConfigProvider.__the_config = ConfigParser(config_path).parse()
        return ConfigProvider.__the_config


