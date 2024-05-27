import yaml

class Config:
    """
    Loads and parses the config file
    """
    def __init__(self, config_file_path: str) -> None:
        with open(config_file_path, 'r') as handle:
            config_dict = yaml.safe_load(handle)
        
        for key, value in config_dict.items():
            self.__setattr__(key, value)
