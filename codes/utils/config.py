import json
import os
from addict import Dict
from copy import deepcopy
from time import time
import datetime
import yaml

from codes.utils.util import make_dir, get_current_commit_id

def _read_config(config_id=None):
    '''
    Method to read the config file and return as a dict
    '''
    print(os.path.dirname(os.path.realpath(__file__)))
    path =  os.path.dirname(os.path.realpath(__file__)).split('codes')[0]
    config_name = "config.yaml"
    if (config_id):
        config_name = "{}.yaml".format(config_id)
    return yaml.load(open(os.path.join(path, 'config', config_name)))


def _read_sample_config():
    '''
    Method to read the config file and return as a dict
    :return:
    '''
    path = os.path.abspath(os.pardir).split('codes')[0]
    return yaml.load(open(os.path.join(path, 'config', 'sample.config.yaml')))


def _get_boolean_value(value):
    if (type(value) == bool):
        return value
    elif (value.lower() == "true"):
        return True
    else:
        return False


def get_config(config_id=None):
    '''Method to prepare the config for all downstream tasks'''

    config = get_base_config(config_id)

    return config


def get_sample_config():
    '''Method to prepare the config for all downstream tasks'''

    config = get_sample_base_config()

    return config


def _post_process(config):
    # Post Processing on the config addict

    config.general = _post_process_general_config(deepcopy(config.general))
    config.dataset = _post_process_dataset_config(deepcopy(config.dataset), config.general)
    config.model = _post_process_model_config(deepcopy(config.model), config)
    config.log = _post_process_log_config(deepcopy(config.log), config.general)
    config.plot = _post_process_plot_config(deepcopy(config.plot), config.general)

    return config


def _post_process_general_config(general_config):
    # Method to post process the general section of the config

    if "seed" not in general_config:
        general_config.seed = 42
    else:
        general_config.seed = int(general_config.seed)

    if ("base_path" not in general_config) or (general_config.base_path == ""):
        general_config.base_path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
        print(general_config.base_path)

    if ("device" not in general_config) or (general_config.device == ""):
        general_config.device = "cpu"

    if ("id" not in general_config) or (general_config.id == ""):
        general_config.id = str(int(time()))
    else:
        general_config.id = str(general_config.id)

    if ("commit_id" not in general_config) or (general_config.commit_id == ""):
        general_config.commit_id = get_current_commit_id()

    if ("env" not in general_config) or (general_config.env == ""):
        general_config.env="prod"

    if ("date" not in general_config) or (not general_config.date):
        now = datetime.datetime.now()
        general_config.date = now.strftime("%Y-%m-%d %H:%M")

    return general_config


def _post_process_dataset_config(dataset_config, general_config):
    # Method to post process the dataset section of the config
    if ("base_path" not in dataset_config) or (dataset_config.base_path == ""):
        dataset_config.base_path = os.path.join(general_config.base_path, "data")

    if ("name" not in dataset_config) or (dataset_config.name == ""):
        dataset_config.name = "family"

    if ("should_preprocess" not in dataset_config) or (dataset_config.should_preprocess == ""):
        dataset_config.should_preprocess = True
    else:
        dataset_config.should_preprocess = _get_boolean_value(dataset_config.should_preprocess)

    return dataset_config


def _post_process_model_config(model_config, config):
    # Method to post process the net section of the config

    general_config = config.general
    dataset_config = config.dataset

    default_params = {"name": "baseline1",
                      "batch_size": 100,
                      "num_epochs": 1000,
                      "persist_per_epoch": -1,
                      "early_stopping_patience": 1,
                      "save_dir": "",
                      "should_load_model": False,
                      "dropout_probability": 0.0,
                      "embedding": {},
                      "optimiser": {},
                      "early_stopping": {}
                      }

    for key in default_params:
        if key not in model_config:
            model_config[key] = default_params[key]

    if (model_config.save_dir == ""):
        model_config.save_dir = os.path.join(general_config.base_path, "net", general_config.id)
    elif (model_config.save_dir[0] != "/"):
        model_config.save_dir = os.path.join(general_config.base_path, model_config.save_dir)

    make_dir(path=model_config.save_dir)

    model_config.load_path = os.path.join(general_config.base_path,
                                          "net", general_config.id)


    for key in ["should_load_model"]:
        model_config[key] = _get_boolean_value(model_config[key])

    model_config.early_stopping = _post_process_early_stooping_config(model_config.early_stopping)
    model_config.embedding = _post_process_embedding_config(deepcopy(model_config.embedding),
                                                            general_config, dataset_config)
    model_config.optimiser = _post_process_optimiser_config(deepcopy(model_config.optimiser))

    model_config.decoder = _post_process_decoder_config(deepcopy(model_config.decoder), model_config.encoder)

    return model_config

def _post_process_decoder_config(decoder_model_config, encoder_model_config):
    decoder_model_config.hidden_dim = encoder_model_config.hidden_dim
    if(encoder_model_config.bidirectional):
        decoder_model_config.hidden_dim = 2*decoder_model_config.hidden_dim
    return decoder_model_config

def _post_process_early_stooping_config(early_stopping_config):
    #     Method to post process the early stopping config
    default_params = {
    "patience": 3,
    "metric_to_track": "loss"
    }
    for key in default_params:
        if key not in early_stopping_config:
            early_stopping_config[key] = default_params[key]

    return early_stopping_config

def _post_process_optimiser_config(optimiser_config):
    # Method to post process the optimiser section of the net config

    default_params = {
        "name": "adam",
        "learning_rate": 0.001,
        "scheduler_type": "exp",
        "scheduler_gamma": 1.0,
        "scheduler_patience": 10,
        "l2_penalty": 0.0}

    for key in default_params:
        if key not in optimiser_config:
            optimiser_config[key] = default_params[key]

    return optimiser_config


def _post_process_embedding_config(embedding_config, general_config, dataset_config):
    # Method to post process the embbedding section of the net config

    default_params = {
        "dim": 100,
        "should_use_pretrained_embedding": "true",
        "should_finetune_embedding": "true",
        "pretrained_embedding_path": "w2v/w2v_sst.txt",

    }

    for key in default_params:
        if key not in embedding_config:
            embedding_config[key] = default_params[key]

    for key in ["should_finetune_embedding", "should_use_pretrained_embedding"]:
        embedding_config[key] = _get_boolean_value(embedding_config[key])

    if (embedding_config.pretrained_embedding_path == ""):
        file_name = "{}_embedding_file".format(dataset_config.name)
        embedding_config.pretrained_embedding_path = os.path.join(dataset_config.base_path, dataset_config.name,
                                                              file_name)

    elif (embedding_config.pretrained_embedding_path[0] != "/"):
        embedding_config.pretrained_embedding_path = os.path.join(general_config.base_path,
                                                              "net",
                                                                  embedding_config.pretrained_embedding_path)

    return embedding_config


def _post_process_plot_config(plot_config, general_config):
    # Method to post process the plot section of the config
    if ("base_path" not in plot_config) or (plot_config.base_path == ""):
        plot_config.base_path = os.path.join(general_config.base_path,
                                             "plots", general_config.id)
        make_dir(path=plot_config.base_path)

    return plot_config


def _post_process_log_config(log_config, general_config):
    # Method to post process the log section of the config

    if ("file_path" not in log_config) or (log_config.file_path == ""):
        log_config.file_path = os.path.join(general_config.base_path,
                                            "logs", general_config.id)
        make_dir(path=log_config.file_path)
        log_config.file_path = os.path.join(log_config.file_path, "log.txt")

    log_config.dir = log_config.file_path.rsplit("/", 1)[0]

    key = "mongo_host"
    if (key not in log_config or log_config[key] == ""):
        log_config[key] = "127.0.0.1"

    key = "mongo_port"
    if (key not in log_config or log_config[key] == ""):
        log_config[key] = "8092"

    key = "mongo_db"
    if (key not in log_config or log_config[key] == ""):
        log_config[key] = "graphsum"

    return log_config


def get_base_config(config_id=None):
    # Returns the bare minimum config (addict) needed to run the experiment
    config_dict = _read_config(config_id)
    return Dict(config_dict)


def get_sample_base_config():
    # Reads the sample config and returns as addict
    config_dict = _read_sample_config()
    return Dict(config_dict)


if __name__ == "__main__":
    print(get_config())