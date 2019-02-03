import random

import numpy as np
import torch

from sacred.observers import FileStorageObserver


from codes.experiments.hier_classifier import run_experiment
from codes.utils.config import get_config
from codes.utils.util import set_seed
from codes.utils.argument_parser import argument_parser
import os
from addict import Dict
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from sacred import Experiment
ex = Experiment('dummy_name')


@ex.main
def start(_config, _run):
    config = Dict(_config)
    set_seed(seed=config.seed)
    run_experiment(config, _run)

if __name__ == '__main__':
    config_id = argument_parser()
    print(config_id)
    config = get_config(config_id=config_id)
    ex.add_config(config)
    options = {}
    options['--name'] = 'exp_{}'.format(config_id)
    if config.logging.use_mongo:
        options['--mongo_db'] = '{}:{}:{}'.format(config.log.mongo_host,
                                        config.log.mongo_port,
                                        config.log.mongo_db)
    else:
        base_path = str(os.path.dirname(os.path.realpath(__file__)).split('/codes')[0])
        log_path = os.path.join(base_path, config.logging.dir)
        ex.observers.append(FileStorageObserver.create(log_path))
    ex.run(options=options)