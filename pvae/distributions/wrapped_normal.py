import logging

import torch
from torch.nn import functional as F
from torch.distributions import Normal, Independent
from numbers import Number
from torch.distributions.utils import _standard_normal, broadcast_all
import geoopt

logger = logging.getLogger(__name__)


class WrappedNormal(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(self, loc, scale, manifold: geoopt.PoincareBall, validate_args=None, softplus=False):
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold
        if self.manifold.c != 0:
            self.manifold.assert_check_point_on_manifold(self.loc)
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.manifold.dim])
        super(WrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        v = self.scale * _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        self.manifold.assert_check_vector_on_tangent(self.manifold.zero, v)
        v = v / self.manifold.lambda_x(self.manifold.zero, keepdim=True)
        u = self.manifold.transp(self.manifold.zero, self.loc, v)
        z = self.manifold.expmap(self.loc, u)
        return z

    def log_prob(self, x):
        """
        x.shape: [K, batch_size, manifold.dim]
        returns: [K, batch_size, 1]
        """
        # logger.debug("x.shape: %s", x.shape)
        shape = x.shape
        loc = self.loc.unsqueeze(0).expand(x.shape[0], *self.batch_shape, self.manifold.coord_dim)
        if len(shape) < len(loc.shape): x = x.unsqueeze(1)
        v = self.manifold.logmap(loc, x)
        v = self.manifold.transp(loc, self.manifold.zero, v)
        u = v * self.manifold.lambda_x(self.manifold.zero, keepdim=True)
        norm_pdf = Normal(torch.zeros_like(self.scale), self.scale).log_prob(u).sum(-1, keepdim=True)
        logdetexp = self.manifold.logdetexp(loc, x, keepdim=True)
        result = norm_pdf - logdetexp
        # logger.debug("result.shape: %s", result.shape)
        try:
            assert result.shape == x.shape[:-1] + (1,)
        except:
            logger.error("x.shape: %s", x.shape)
            logger.error("loc.shape: %s", loc.shape)
            logger.error("v.shape: %s", v.shape)
            logger.error("u.shape: %s", u.shape)
            logger.error("norm_pdf.shape: %s", norm_pdf.shape)
            logger.error("logdetexp.shape: %s", logdetexp.shape)
            logger.error("result.shape: %s", result.shape)
            raise
        return result


class WrappedNormalDifferentLogProb(WrappedNormal):
    def log_prob(self, zs: torch.Tensor):
        # zs: [K, batch_size, manifold.dim]
        loc, scale, manifold = self.loc, self.scale, self.manifold
        d = loc.shape[-1]
        lambda_mu_c = manifold.lambda_x(loc)
        logmap_mu_zs = manifold.logmap(loc, zs)
        assert logmap_mu_zs.shape == zs.shape
        lambda_mu_c_view = lambda_mu_c.view(1, lambda_mu_c.numel(), 1)
        try:
            x0 = lambda_mu_c_view * logmap_mu_zs
        except:
            logger.error("loc.shape: %s", loc.shape)
            logger.error("zs.shape: %s", zs.shape)
            logger.error("lambda_mu_c.shape: %s", lambda_mu_c.shape)
            logger.error("lambda_mu_c_view.shape: %s", lambda_mu_c_view.shape)
            logger.error("logmap_mu_zs.shape: %s", logmap_mu_zs.shape)
            raise
        # logger.debug("x0.shape: %s", x0.shape)
        x1 = Normal(torch.zeros_like(loc), scale).log_prob(x0).sum(-1, keepdim=True)
        x2 = manifold.c.sqrt() + manifold.dist(loc, zs, keepdim=True)
        x3 = torch.sinh(x2)
        try:
            result = x1 + (d - 1) * (x2.log() - x3.log())
            assert result.shape == zs.shape[:-1] + (1,)
        except:
            logger.error("zs.shape: %s", zs.shape)
            logger.error("x1.shape: %s", x1.shape)
            logger.error("x2.shape: %s", x2.shape)
            logger.error("x3.shape: %s", x3.shape)
            logger.error("new_var.shape: %s", result.shape)
            raise
        return result
