import os

import json
from easydict import EasyDict
from pprint import pprint

from utils.dirs import create_dirs


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)

            if 'current_epoch' not in config or not config['current_epoch']:
                config.current_epoch = 0

            if 'num_log_samples' not in config or not config['num_log_samples']:
                config.num_log_samples = 32
            
            if 'optimizer' not in config or not config['optimizer']:
                config.optimizer = 'SGD'

            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(json_file):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(json_file)
    print(" THE Configuration of your experiment ..")
    pprint(config)

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    # create some important directories to be used for that experiment.
    config.summary_dir = os.path.join("experiments", config.exp_name,
                                      "summaries/")
    config.checkpoint_dir = os.path.join("experiments", config.exp_name,
                                         "checkpoints/")
    create_dirs([config.summary_dir, config.checkpoint_dir])

    return config
