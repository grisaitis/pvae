# Base VAE class definition

import logging
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from pvae.utils import get_mean_param
from pvae.vis_other import plot_posterior_means_for_df

logger = logging.getLogger(__name__)


class VAE(nn.Module):
    def __init__(self, prior_dist, posterior_dist, likelihood_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params
        self.data_size = params.data_size
        self.prior_std = params.prior_std

        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        raise NotImplementedError

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            px_z_params = self.dec(self.pz(*self.pz_params).sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample(torch.Size([1])).squeeze(0))

        return get_mean_param(px_z_params)

    def forward(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std_scale)

    def init_last_layer_bias(self, dataset): pass

    def map_class_labels_to_1d(self, class_labels: np.ndarray) -> np.ndarray:
        assert class_labels.ndim == 1, class_labels.shape
        return class_labels

    def plot_posterior_means(self, data_loader, runPath, epoch):
        self.eval()
        model_device = next(self.parameters()).device
        with torch.no_grad():
            # for each batch in the data_loader, compute the posterior means
            # and store the means and class labels in a DataFrame
            posterior_means = []
            class_labels = []
            for i_batch, (data, labels) in enumerate(data_loader):
                logger.debug("encoding batch %d", i_batch)
                data = data.to(model_device)
                pz_x_params = self.enc(data)
                pz_x_means = get_mean_param(pz_x_params)
                posterior_means.append(pz_x_means)
                class_labels.append(labels)
        means = torch.cat(posterior_means)
        class_labels_tensor = torch.cat(class_labels)
        try:
            data_numpy = {
                "z0": means[:, 0].numpy(),
                "z1": means[:, 1].numpy(),
                "class_label": class_labels_tensor.numpy()
            }
        except TypeError:
            data_numpy = {
                "z0": means[:, 0].cpu().numpy(),
                "z1": means[:, 1].cpu().numpy(),
                "class_label": class_labels_tensor.cpu().numpy()
            }
        data_numpy["class_label"] = self.map_class_labels_to_1d(data_numpy["class_label"])
        data_numpy["class_label"] = data_numpy["class_label"].astype(str)
        logger.debug("shape of z0, z1, class_label: %s, %s, %s", data_numpy["z0"].shape, data_numpy["z1"].shape, data_numpy["class_label"].shape)
        df_means = pd.DataFrame(data_numpy)
        if self.params.manifold == 'PoincareBall':
            radius = self.params.c**-0.5
            axis_range = [-(radius), radius]
        else:
            axis_range = [-5, 5]
        fig = plot_posterior_means_for_df(df_means, axis_range)
        filepath = "{}/posterior_means_{:03d}.png".format(runPath, epoch)
        logger.debug("writing posterior means plot to %s", Path(filepath).resolve())
        fig.write_image(filepath, scale=2)
