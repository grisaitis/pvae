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
from pathlib import Path

import torch
from torch import optim
import numpy as np

from utils import Logger, Timer, save_model, save_vars, probe_infnan
import objectives
import models
from models.utils import load_model


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

    # Initialise model, dataset loader
    logger.debug("Initialising model")
    model = load_model(args.run_path, device)
    if model.params.model == "tree":
        logger.debug("Loading tree model dataloaders")
        train_loader = torch.load(args.run_path + '/dataloader_train.pt')
        test_loader = torch.load(args.run_path + '/dataloader_test.pt')
    else:
        train_loader, test_loader = model.getDataLoaders(model.params.batch_size, True, device, *model.params.data_params)
    fig = model.plot_posterior_means(test_loader)
    filepath = "{}/posthoc_model_test_posterior_means.png".format(args.run_path)
    logger.debug("writing posterior means plot...")
    fig.write_image(filepath, scale=2)
    logger.debug("saved image to %s", Path(filepath).resolve())
