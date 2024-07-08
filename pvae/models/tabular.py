import logging
import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.utils.data
from torch.utils.data import DataLoader

import math
from sklearn.model_selection._split import _validate_shuffle_split

from pvae.utils import get_mean_param
from .vae import VAE
from pvae.vis import array_plot

from pvae.distributions import RiemannianNormal, WrappedNormal, WrappedNormalDifferentLogProb
from torch.distributions import Normal
from pvae import manifolds
from .architectures import EncLinear, DecLinear, EncWrapped, DecWrapped, EncMob, DecMob, DecGeo, DecLinearWithScale
from pvae.datasets import SyntheticDataset, CSVDataset, JerbyArnonDataset

logger = logging.getLogger(__name__)


class Tabular(VAE):
    """ Derive a specific sub-class of a VAE for tabular data. """
    def __init__(self, params):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        super(Tabular, self).__init__(
            eval(params.prior),           # prior distribution
            eval(params.posterior),       # posterior distribution
            dist.Normal,                  # likelihood distribution
            eval('Enc' + params.enc)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim, params.prior_iso),
            eval('Dec' + params.dec)(manifold, params.data_size, getattr(nn, params.nl)(), params.num_hidden_layers, params.hidden_dim),
            params
        )
        self.manifold = manifold
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)
        self.modelName = 'Tabular'

    @property
    def pz_params(self):
        return self._pz_mu.mul(1), F.softplus(self._pz_logvar).div(math.log(2)).mul(self.prior_std), self.manifold

    def generate(self, runPath, epoch):
        N, K = 10, 1
        _, _, samples = super(Tabular, self).generate(N, K)
        array_plot([samples.data.cpu()], '{}/gen_samples_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        recon = super(Tabular, self).reconstruct(data)
        array_plot([data.data.cpu(), recon.data.cpu()], '{}/reconstruct_{:03d}.png'.format(runPath, epoch))


class Tree(Tabular):
    """ Derive a specific sub-class of a VAE for tree data. """
    def __init__(self, params):
        super(Tree, self).__init__(params)

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        dataset = self._create_dataset(args)
        train_loader, test_loader = self._create_dataloaders(batch_size, shuffle, device, dataset)
        return train_loader, test_loader

    def _create_dataloaders(self, batch_size, shuffle, device, dataset):
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        logger.debug("Train size: %d, Test size: %d", n_train, n_test)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
        return train_loader,test_loader

    def _create_dataset(self, args):
        logger.info("Load training data with data_size %s", self.data_size)
        print('Load training data...')
        dataset = SyntheticDataset(*self.data_size, *map(lambda x: float(x), args))
        logger.debug("Dataset size: %d", len(dataset))
        return dataset

    def _map_class_labels_to_1d(self, class_labels: np.ndarray):
        # shape is like (n, k)
        # map the one-hot encodings to a single integer
        # res = class_labels.argmax(axis=1)
        # map each row of class_labels to a string
        res = np.array(['-'.join([str(int(i)) for i in row[:3] if i > 0]) for row in class_labels])
        assert res.shape == (class_labels.shape[0],), res.shape
        return res


class CSV(Tabular):
    """ Derive a specific sub-class of a VAE for tabular data loaded via a cvs file. """
    def __init__(self, params):
        super(CSV, self).__init__(params)

    def getDataLoaders(self, batch_size, shuffle, device, *args):
        split_generator = torch.Generator().manual_seed(self.params.seed)
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        print('Load training data...')
        dataset = CSVDataset(*args)
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test], generator=split_generator)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
        return train_loader, test_loader


class Rnaseq(Tabular):
    # def __init__(self, params):
        # super(Rnaseq, self).__init__(params)
        # self.px_z = dist.NegativeBinomial

    @staticmethod
    def getDataLoaders(batch_size, shuffle, device, *args):
        dataset = JerbyArnonDataset.from_csv(
            "data/jerby_arnon/GSE115978_tpm.csv",
            "data/jerby_arnon/GSE115978_cell.annotations.csv",
            standardize_rnaseq=True,
        )
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        logger.debug("random splitting dataset into %s train and %s test samples", n_train, n_test)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
        return train_loader, test_loader

    def _process_label_batches(self, label_batches) -> typing.List[str]:
        return [
            cell_type
            for label_batch in label_batches
            for cell_type in label_batch[1]
        ]

    def generate(self, runPath, epoch):
        pass

    def reconstruct(self, data, runPath, epoch):
        pass


class RnaseqNegativeBinomial(Rnaseq):
    def __init__(self, params):
        super(RnaseqNegativeBinomial, self).__init__(params)
        self.px_z = dist.Binomial
        # dataset = JerbyArnonDataset.from_csv(
        #     "data/jerby_arnon/GSE115978_counts.csv",
        #     "data/jerby_arnon/GSE115978_cell.annotations.csv",
        #     standardize_rnaseq=False
        # )
        # device = torch.device("cuda" if self.params.cuda else "cpu")
        # self.gene_means = torch.Tensor(dataset.df_rnaseq.mean(axis=1).values).to(device)
        # self.gene_stds = torch.Tensor(dataset.df_rnaseq.std(axis=1).values).to(device)
        # logger.debug("devices: gene_means=%s, gene_stds=%s", self.gene_means.device, self.gene_stds.device)

    @staticmethod
    def getDataLoaders(batch_size, shuffle, device, *args):
        dataset = JerbyArnonDataset.from_csv(
            "data/jerby_arnon/GSE115978_counts.csv",
            "data/jerby_arnon/GSE115978_cell.annotations.csv",
            standardize_rnaseq=True,
        )
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == "cuda" else {}
        n_train, n_test = _validate_shuffle_split(len(dataset), test_size=None, train_size=0.7)
        logger.debug("random splitting dataset into %s train and %s test samples", n_train, n_test)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, shuffle=False, **kwargs)
        return train_loader, test_loader

    def forward_vae(self, x, K=1):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    def forward(self, x: torch.Tensor, K=1):
        total_counts = torch.ones_like(x) * 400_000
        # logger.debug("some total_counts values are\n%s", total_counts[:10])
        # total_counts = total_counts.type(torch.int32)
        # logger.debug("some total_counts values after casting are\n%s", total_counts[:10])
        
        # try:
        #     x = (x - self.gene_means) / self.gene_stds
        # except:
        #     logger.error("devices: x=%s, gene_means=%s, gene_stds=%s", x.device, self.gene_means.device, self.gene_stds.device)
        #     logger.error("x shape: %s", x.shape)
        #     logger.error("gene_means shape: %s", self.gene_means.shape)
        #     logger.error("gene_stds shape: %s", self.gene_stds.shape)
        #     raise
        # logger.debug("x.mean(dim=-2) first genes are\n%s", x.mean(dim=-2)[..., :8])
        # logger.debug("x.std(dim=-2) first genes are\n%s", x.std(dim=-2)[..., :8])
        
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        # logger.debug("zs first values are\n%s", zs[0, :8])
        px_z_logits = get_mean_param(self.dec(zs))
        # logger.debug("px_z_logits first values are\n%s", px_z_logits[0, :5, :8])
        px_z = self.px_z(total_counts, logits=px_z_logits)

        # log_probs = px_z.log_prob(x)
        # logger.debug("log_probs shape: %s", log_probs.shape)
        # logger.debug("log_probs mean=%s, std=%s", log_probs.mean().item(), log_probs.std().item())

        return qz_x, px_z, zs

    # def init_last_layer_bias(self, train_loader):
    #     logger.debug("initializing last layer bias")
    #     data_size = self.params.data_size
    #     from numpy import prod
    #     if not hasattr(self.dec.fc31, "bias"):
    #         return
    #     with torch.no_grad():
    #         p = torch.zeros(prod(data_size), device=self._pz_mu.device)
    #         N = 0
    #         for i, (data, _) in enumerate(train_loader):
    #             data = data.to(self._pz_mu.device)
    #             B = data.size(0)
    #             N += B
    #             p += data.view(-1, prod(data_size)).sum(0)
    #         p /= N
    #         p += 1e-4
    #         self.dec.fc31.bias.set_(p.log() - (1 - p).log())
    #     logger.debug("last layer bias initialized")
