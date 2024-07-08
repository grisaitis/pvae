import logging
import torch
import torch.distributions as dist
from numpy import prod
from pvae.utils import has_analytic_kl, log_mean_exp
import pvae.distributions
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def my_analytic_kl(dist_1: dist.Distribution, dist_2: dist.Distribution):
    # KL divergence of dist_1 from dist_2
    B, D = dist_1.loc.shape
    def get_params(p: dist.Distribution):
        if isinstance(p, dist.Normal):
            return p.loc, p.scale
        elif isinstance(p, pvae.distributions.WrappedNormal):
            return p.manifold.logmap0(p.loc), p.scale
        else:
            raise NotImplementedError
    mu_1, scale_1 = get_params(dist_1)
    mu_2, scale_2 = get_params(dist_2)
    x1 = (mu_2 - mu_1)**2 / scale_2**2
    x2 = scale_1**2 / scale_2**2
    x3 = 2.0 * (torch.log(scale_2) - torch.log(scale_1))
    kl_each_dim = 0.5 * (x1 + x2 + x3 - 1)
    result = kl_each_dim.unsqueeze(0).sum(-1)
    try:
        assert result.shape == (1, B,)
    except:
        logger.error("result.shape: %s", result.shape)
        logger.error("K: %s, B: %s", K, B)
        logger.error("mu_1.shape: %s", mu_1.shape)
        logger.error("scale_1.shape: %s", scale_1.shape)
        logger.error("mu_2.shape: %s", mu_2.shape)
        logger.error("scale_2.shape: %s", scale_2.shape)
        logger.error("x1.shape: %s", x1.shape)
        logger.error("x2.shape: %s", x2.shape)
        logger.error("x3.shape: %s", x3.shape)
        raise
    return result


def vae_objective(model, x, K=1, beta=1.0, components=False, analytical_kl=False, **kwargs):
    """Computes E_{p(x)}[ELBO] """
    # logger.debug("x.shape: %s", x.shape)
    qz_x, px_z, zs = model(x, K)
    _, B, D = zs.size()
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpx_z = px_z.log_prob(x.expand(px_z.batch_shape)).view(flat_rest).sum(-1)

    pz = model.pz(*model.pz_params)
    # if analytical_kl:
    #     kld = my_analytic_kl(qz_x, pz)
    # else:
    kld = dist.kl_divergence(qz_x, pz).unsqueeze(0).sum(-1) if \
        has_analytic_kl(type(qz_x), model.pz) and analytical_kl else \
        qz_x.log_prob(zs).sum(-1) - pz.log_prob(zs).sum(-1)    

    obj = -lpx_z.mean(0).sum() + beta * kld.mean(0).sum()
    return (qz_x, px_z, lpx_z, kld, obj) if components else obj

def _iwae_objective_vec(model, x, K):
    """Helper for IWAE estimate for log p_\theta(x) -- full vectorisation."""
    qz_x, px_z, zs = model(x, K)
    flat_rest = torch.Size([*px_z.batch_shape[:2], -1])
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x.expand(zs.size(0), *x.size())).view(flat_rest).sum(-1)
    lqz_x = qz_x.log_prob(zs).sum(-1)
    obj = lpz.squeeze(-1) + lpx_z.view(lpz.squeeze(-1).shape) - lqz_x.squeeze(-1)
    return -log_mean_exp(obj).sum()


def iwae_objective(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    Appropriate negation (for minimisation) happens in the helper
    """
    split_size = int(x.size(0) / (K * prod(x.size()) / (1e8)))  # rough heuristic
    split_size = max(1, split_size)
    if split_size >= x.size(0):
        obj = _iwae_objective_vec(model, x, K)
    else:
        obj = 0
        for bx in x.split(split_size):
            obj = obj + _iwae_objective_vec(model, bx, K)
    return obj
