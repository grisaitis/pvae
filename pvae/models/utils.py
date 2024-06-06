import argparse
import json
from logging import Logger
import torch
from models.vae import VAE
from pvae import models

logger = Logger(__name__)


def load_model(run_path: str, device: torch.device) -> VAE:
    """
    Load a saved model from the given run path.
    Not classmethod, because class isn't known before loading arg files.
    """
    # fail fast - error out if no model exists
    model_state = torch.load(run_path + '/model.rar', map_location=device)
    run_path_args = load_model_params(run_path)
    modelC = getattr(models, 'VAE_{}'.format(run_path_args.model))
    logger.debug("Initialising model with class %s", modelC)
    model = modelC(run_path_args)
    model.load_state_dict(model_state)
    return model


def load_model_params(run_path: str) -> argparse.Namespace:
    run_path_args_json = run_path + '/args.json'
    logger.debug("Loading args from %s", run_path_args_json)
    with open(run_path_args_json, 'r') as f:
        run_path_args_dict = json.load(f)
    run_path_args = argparse.Namespace(**run_path_args_dict)
    return run_path_args
