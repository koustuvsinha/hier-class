## Model utils to load and save models
import torch
import os
from os.path import dirname, abspath
import json
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def create_save_dir(exp_name):
    # Create directory if not present in `saved` in the experiment name
    # going up 3 levels
    parent_dir = dirname(dirname(dirname(abspath(__file__))))
    save_path_base = parent_dir + '/saved/' + exp_name
    if not os.path.exists(save_path_base):
        os.makedirs(save_path_base)
    return save_path_base

def save_model(model, epoch=0, step=0, exp_name='', params=None):
    """ Save model, model params
    Check if model params serializable type then save
    """

    save_path_base = create_save_dir(exp_name)
    save_path = save_path_base + '/' + \
                params['save_name'].format(epoch, step)
    logging.info("Saving model in {}".format(save_path))
    if hasattr(model, "save_state_dict"):
        model.save_state_dict(save_path)
    else:
        torch.save(
            model.state_dict(),
            save_path
        )
    logging.info("Model saved, now saving parameters")
    # nix params which are not json serializable
    to_save = {}
    for key, val in params.items():
        if is_jsonable(val):
            to_save[key] = val
    json.dump(to_save, open(save_path_base + '/parameters.json', 'w'))
    logging.info("Saved model and params")