import logging
import sys
sys.path.append(".")
sys.path.append("..")
import os
import datetime
import json
import argparse
from tempfile import mkdtemp
from collections import defaultdict
import subprocess
import math
import torch
from torch import optim
import numpy as np

from utils import Logger, Timer, save_model, save_vars, probe_infnan
import objectives
import models

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

### General
parser.add_argument('--run-path', type=str)

### Technical
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')


if __name__ == '__main__':
    logging.basicConfig(
        level="DEBUG",
        # format="%(asctime)s %(name)s %(levelname)s %(message)s",
        format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger.debug("Parsing arguments")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    logger.debug("Device: %s", device)

    # load args, update some
    run_path_args_json = args.run_path + '/args.json'
    logger.debug("Loading args from %s", run_path_args_json)
    with open(run_path_args_json, 'r') as f:
        run_path_args_dict = json.load(f)
    run_path_args = argparse.Namespace(**run_path_args_dict)
    run_path_args.cuda = args.cuda

    # Initialise model, dataset loader
    logger.debug("Initialising model")
    modelC = getattr(models, 'VAE_{}'.format(run_path_args.model))
    model = modelC(run_path_args).to(device)
    train_loader, test_loader = model.getDataLoaders(run_path_args.batch_size, True, device, *run_path_args.data_params)
    
    model_state_path = args.run_path + '/model.rar'
    with open(model_state_path, 'rb') as f:
        model_state_dict = torch.load(f)
    model.load_state_dict(model_state_dict)
    model.eval()
    # print(model)
    # print(model.__dict__.keys())
    model.plot_posterior_means(test_loader, args.run_path, epoch=0)
